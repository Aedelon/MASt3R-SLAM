# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Sim3 Lie group implementation in MLX."""

from __future__ import annotations

import mlx.core as mx

from mlx_mast3r_slam.liegroups.so3 import SO3, _cross
from mlx_mast3r_slam.liegroups.se3 import SE3


class Sim3:
    """Similarity Group Sim(3) - 3D similarity transformations (rotation + translation + scale).

    Internally represented as [tx, ty, tz, qx, qy, qz, qw, s] where s is the scale.
    """

    tangent_dim = 7
    embedded_dim = 8

    def __init__(self, data: mx.array) -> None:
        """Initialize Sim3 from [translation, quaternion, scale] data.

        Args:
            data: Array of shape [..., 8] with [tx, ty, tz, qx, qy, qz, qw, s]
        """
        self.data = data

    @property
    def translation(self) -> mx.array:
        """Get translation component."""
        return self.data[..., :3]

    @property
    def rotation(self) -> SO3:
        """Get rotation component as SO3."""
        return SO3(self.data[..., 3:7])

    @property
    def scale(self) -> mx.array:
        """Get scale component."""
        return self.data[..., 7:8]

    @classmethod
    def identity(
        cls, batch_shape: tuple[int, ...] = (), dtype: mx.Dtype = mx.float32
    ) -> Sim3:
        """Create identity transformation(s).

        Args:
            batch_shape: Shape for batch dimensions
            dtype: Data type

        Returns:
            Identity Sim3 element(s)
        """
        shape = batch_shape + (8,)
        data = mx.zeros(shape, dtype=dtype)
        # qw = 1 for identity quaternion
        data = data.at[..., 6].add(1.0)
        # scale = 1 for identity
        data = data.at[..., 7].add(1.0)
        return cls(data)

    @classmethod
    def from_rotation_translation_scale(
        cls, rotation: SO3, translation: mx.array, scale: mx.array
    ) -> Sim3:
        """Create Sim3 from rotation, translation and scale.

        Args:
            rotation: SO3 rotation
            translation: Translation vector [..., 3]
            scale: Scale factor [..., 1]

        Returns:
            Sim3 element
        """
        data = mx.concatenate([translation, rotation.data, scale], axis=-1)
        return cls(data)

    @classmethod
    def from_se3(cls, se3: SE3, scale: mx.array | None = None) -> Sim3:
        """Create Sim3 from SE3 with optional scale.

        Args:
            se3: SE3 transformation
            scale: Scale factor (default 1.0)

        Returns:
            Sim3 element
        """
        if scale is None:
            scale = mx.ones(se3.shape + (1,), dtype=se3.data.dtype)
        data = mx.concatenate([se3.data, scale], axis=-1)
        return cls(data)

    def to_se3(self) -> SE3:
        """Convert to SE3 (ignoring scale).

        Returns:
            SE3 element
        """
        return SE3(self.data[..., :7])

    @classmethod
    def exp(cls, tau: mx.array) -> Sim3:
        """Exponential map from sim3 (tangent space) to Sim3.

        Args:
            tau: Tangent vector [..., 7] with [v, omega, sigma] (translation, rotation, log-scale)

        Returns:
            Sim3 element
        """
        v = tau[..., :3]  # Translation part
        omega = tau[..., 3:6]  # Rotation part
        sigma = tau[..., 6:7]  # Log-scale

        theta_sq = mx.sum(omega * omega, axis=-1, keepdims=True)
        theta = mx.sqrt(theta_sq + 1e-10)

        # Small angle check
        small_angle = theta_sq < 1e-8

        # Scale
        s = mx.exp(sigma)

        # For Sim3, the translation computation involves scale
        # W matrix (generalized jacobian)
        A = mx.where(
            small_angle,
            1.0 - theta_sq / 6.0,
            mx.sin(theta) / theta,
        )
        B = mx.where(
            small_angle,
            0.5 - theta_sq / 24.0,
            (1.0 - mx.cos(theta)) / theta_sq,
        )

        # Simplified: use SE3-like translation with scale adjustment
        omega_cross_v = _cross(omega, v)
        C = mx.where(
            small_angle,
            1.0 / 6.0 - theta_sq / 120.0,
            (1.0 - A) / theta_sq,
        )
        t = v + B * omega_cross_v + C * _cross(omega, omega_cross_v)

        # Rotation part
        rotation = SO3.exp(omega)

        return cls.from_rotation_translation_scale(rotation, t, s)

    def log(self) -> mx.array:
        """Logarithmic map from Sim3 to sim3 (tangent space).

        Returns:
            Tangent vector [..., 7]
        """
        omega = self.rotation.log()
        t = self.translation
        sigma = mx.log(self.scale + 1e-10)

        theta_sq = mx.sum(omega * omega, axis=-1, keepdims=True)
        theta = mx.sqrt(theta_sq + 1e-10)

        small_angle = theta_sq < 1e-8

        # Inverse of W (generalized jacobian)
        half_theta = 0.5 * theta
        A = mx.where(
            small_angle,
            1.0 - theta_sq / 6.0,
            mx.sin(theta) / theta,
        )
        cot_half = mx.where(
            small_angle,
            2.0 / theta - theta / 6.0,
            mx.cos(half_theta) / mx.sin(half_theta + 1e-10),
        )
        B_inv = mx.where(
            small_angle,
            1.0 / 12.0 + theta_sq / 720.0,
            (1.0 - 0.5 * A * cot_half) / theta_sq,
        )

        # W^-1 * t
        omega_cross_t = _cross(omega, t)
        v = t - 0.5 * omega_cross_t + B_inv * _cross(omega, omega_cross_t)

        return mx.concatenate([v, omega, sigma], axis=-1)

    def inv(self) -> Sim3:
        """Inverse transformation.

        Returns:
            Inverse Sim3 element
        """
        R_inv = self.rotation.inv()
        s_inv = 1.0 / (self.scale + 1e-10)
        t_inv = -s_inv * R_inv.act(self.translation)
        return Sim3.from_rotation_translation_scale(R_inv, t_inv, s_inv)

    def __mul__(self, other: Sim3) -> Sim3:
        """Compose transformations.

        T1 * T2: first apply T2, then T1

        Args:
            other: Another Sim3 element

        Returns:
            Composed transformation
        """
        R_new = self.rotation * other.rotation
        s_new = self.scale * other.scale
        t_new = self.translation + self.scale * self.rotation.act(other.translation)
        return Sim3.from_rotation_translation_scale(R_new, t_new, s_new)

    def act(self, point: mx.array) -> mx.array:
        """Transform a 3D point: s * R * p + t.

        Args:
            point: 3D point(s) [..., 3]

        Returns:
            Transformed point(s) [..., 3]
        """
        return self.scale * self.rotation.act(point) + self.translation

    def matrix(self) -> mx.array:
        """Convert to 4x4 transformation matrix.

        Returns:
            Transformation matrix [..., 4, 4] where upper-left 3x3 is s*R
        """
        R = self.rotation.matrix()
        t = self.translation
        s = self.scale

        batch_shape = self.data.shape[:-1]
        T = mx.zeros(batch_shape + (4, 4), dtype=self.data.dtype)

        # Upper-left 3x3: s * R
        T = T.at[..., :3, :3].add(s[..., None] * R)
        T = T.at[..., :3, 3].add(t)
        T = T.at[..., 3, 3].add(1.0)

        return T

    def retr(self, delta: mx.array) -> Sim3:
        """Retraction: update transformation by tangent vector.

        Args:
            delta: Tangent vector [..., 7]

        Returns:
            Updated Sim3 element
        """
        return self * Sim3.exp(delta)

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.data.shape[:-1]

    def __repr__(self) -> str:
        return f"Sim3(shape={self.shape}, data={self.data})"
