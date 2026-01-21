# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""SE3 Lie group implementation in MLX."""

from __future__ import annotations

import mlx.core as mx

from mlx_mast3r_slam.liegroups.so3 import SO3, _cross


class SE3:
    """Special Euclidean Group SE(3) - 3D rigid transformations.

    Internally represented as [tx, ty, tz, qx, qy, qz, qw].
    """

    tangent_dim = 6
    embedded_dim = 7

    def __init__(self, data: mx.array) -> None:
        """Initialize SE3 from [translation, quaternion] data.

        Args:
            data: Array of shape [..., 7] with [tx, ty, tz, qx, qy, qz, qw]
        """
        self.data = data

    @property
    def translation(self) -> mx.array:
        """Get translation component."""
        return self.data[..., :3]

    @property
    def rotation(self) -> SO3:
        """Get rotation component as SO3."""
        return SO3(self.data[..., 3:])

    @classmethod
    def identity(
        cls, batch_shape: tuple[int, ...] = (), dtype: mx.Dtype = mx.float32
    ) -> SE3:
        """Create identity transformation(s).

        Args:
            batch_shape: Shape for batch dimensions
            dtype: Data type

        Returns:
            Identity SE3 element(s)
        """
        shape = batch_shape + (7,)
        data = mx.zeros(shape, dtype=dtype)
        # qw = 1 for identity quaternion
        data = data.at[..., 6].add(1.0)
        return cls(data)

    @classmethod
    def from_rotation_translation(cls, rotation: SO3, translation: mx.array) -> SE3:
        """Create SE3 from rotation and translation.

        Args:
            rotation: SO3 rotation
            translation: Translation vector [..., 3]

        Returns:
            SE3 element
        """
        data = mx.concatenate([translation, rotation.data], axis=-1)
        return cls(data)

    @classmethod
    def exp(cls, tau: mx.array) -> SE3:
        """Exponential map from se3 (tangent space) to SE3.

        Args:
            tau: Tangent vector [..., 6] with [v, omega] (translation, rotation)

        Returns:
            SE3 element
        """
        v = tau[..., :3]  # Translation part
        omega = tau[..., 3:]  # Rotation part

        theta_sq = mx.sum(omega * omega, axis=-1, keepdims=True)
        theta = mx.sqrt(theta_sq + 1e-10)

        # Small angle check
        small_angle = theta_sq < 1e-8

        # Rodrigues formula for V (jacobian of rotation)
        # V = I + (1-cos(theta))/theta^2 * [omega]_x + (theta-sin(theta))/theta^3 * [omega]_x^2
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
        C = mx.where(
            small_angle,
            1.0 / 6.0 - theta_sq / 120.0,
            (1.0 - A) / theta_sq,
        )

        # V * v = v + B * (omega x v) + C * (omega x (omega x v))
        omega_cross_v = _cross(omega, v)
        t = v + B * omega_cross_v + C * _cross(omega, omega_cross_v)

        # Rotation part
        rotation = SO3.exp(omega)

        return cls.from_rotation_translation(rotation, t)

    def log(self) -> mx.array:
        """Logarithmic map from SE3 to se3 (tangent space).

        Returns:
            Tangent vector [..., 6]
        """
        omega = self.rotation.log()
        t = self.translation

        theta_sq = mx.sum(omega * omega, axis=-1, keepdims=True)
        theta = mx.sqrt(theta_sq + 1e-10)

        small_angle = theta_sq < 1e-8

        # Inverse of V
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

        # V^-1 * t = t - 0.5 * (omega x t) + B_inv * (omega x (omega x t))
        omega_cross_t = _cross(omega, t)
        v = t - 0.5 * omega_cross_t + B_inv * _cross(omega, omega_cross_t)

        return mx.concatenate([v, omega], axis=-1)

    def inv(self) -> SE3:
        """Inverse transformation.

        Returns:
            Inverse SE3 element
        """
        R_inv = self.rotation.inv()
        t_inv = -R_inv.act(self.translation)
        return SE3.from_rotation_translation(R_inv, t_inv)

    def __mul__(self, other: SE3) -> SE3:
        """Compose transformations.

        Args:
            other: Another SE3 element

        Returns:
            Composed transformation
        """
        R_new = self.rotation * other.rotation
        t_new = self.translation + self.rotation.act(other.translation)
        return SE3.from_rotation_translation(R_new, t_new)

    def act(self, point: mx.array) -> mx.array:
        """Transform a 3D point.

        Args:
            point: 3D point(s) [..., 3]

        Returns:
            Transformed point(s) [..., 3]
        """
        return self.rotation.act(point) + self.translation

    def matrix(self) -> mx.array:
        """Convert to 4x4 transformation matrix.

        Returns:
            Transformation matrix [..., 4, 4]
        """
        R = self.rotation.matrix()
        t = self.translation

        batch_shape = self.data.shape[:-1]
        T = mx.zeros(batch_shape + (4, 4), dtype=self.data.dtype)

        T = T.at[..., :3, :3].add(R)
        T = T.at[..., :3, 3].add(t)
        T = T.at[..., 3, 3].add(1.0)

        return T

    def retr(self, delta: mx.array) -> SE3:
        """Retraction: update transformation by tangent vector.

        Args:
            delta: Tangent vector [..., 6]

        Returns:
            Updated SE3 element
        """
        return self * SE3.exp(delta)

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.data.shape[:-1]

    def __repr__(self) -> str:
        return f"SE3(shape={self.shape}, data={self.data})"
