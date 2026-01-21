# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""SO3 Lie group implementation in MLX."""

from __future__ import annotations

import mlx.core as mx


def _cross(a: mx.array, b: mx.array) -> mx.array:
    """Cross product of two 3D vectors.

    Args:
        a: Vector [..., 3]
        b: Vector [..., 3]

    Returns:
        Cross product [..., 3]
    """
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return mx.stack([
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ], axis=-1)


class SO3:
    """Special Orthogonal Group SO(3) - 3D rotations.

    Internally represented as unit quaternions [qx, qy, qz, qw] (Hamilton convention).
    """

    tangent_dim = 3
    embedded_dim = 4

    def __init__(self, data: mx.array) -> None:
        """Initialize SO3 from quaternion data [qx, qy, qz, qw].

        Args:
            data: Quaternion array of shape [..., 4]
        """
        self.data = data

    @classmethod
    def identity(
        cls, batch_shape: tuple[int, ...] = (), dtype: mx.Dtype = mx.float32
    ) -> SO3:
        """Create identity rotation(s).

        Args:
            batch_shape: Shape for batch dimensions
            dtype: Data type

        Returns:
            Identity SO3 element(s)
        """
        shape = batch_shape + (4,)
        data = mx.zeros(shape, dtype=dtype)
        # qw = 1 for identity quaternion
        data = data.at[..., 3].add(1.0)
        return cls(data)

    @classmethod
    def exp(cls, omega: mx.array) -> SO3:
        """Exponential map from so3 (tangent space) to SO3.

        Args:
            omega: Rotation vector [..., 3] (axis * angle)

        Returns:
            SO3 element
        """
        theta_sq = mx.sum(omega * omega, axis=-1, keepdims=True)
        theta = mx.sqrt(theta_sq + 1e-10)
        half_theta = 0.5 * theta

        # Small angle approximation
        small_angle = theta_sq < 1e-8

        # Normal case: q = [sin(theta/2) * axis, cos(theta/2)]
        sinc_half = mx.where(
            small_angle,
            0.5 - theta_sq / 48.0,  # Taylor expansion of sin(x)/x at x=0
            mx.sin(half_theta) / theta,
        )
        cos_half = mx.where(
            small_angle,
            1.0 - theta_sq / 8.0,  # Taylor expansion of cos(x) at x=0
            mx.cos(half_theta),
        )

        qxyz = sinc_half * omega
        qw = cos_half
        data = mx.concatenate([qxyz, qw], axis=-1)
        return cls(data)

    def log(self) -> mx.array:
        """Logarithmic map from SO3 to so3 (tangent space).

        Returns:
            Rotation vector [..., 3]
        """
        qxyz = self.data[..., :3]
        qw = self.data[..., 3:4]

        # Ensure qw >= 0 (use canonical quaternion)
        sign = mx.where(qw < 0, -1.0, 1.0)
        qxyz = sign * qxyz
        qw = sign * qw

        norm_qxyz = mx.sqrt(mx.sum(qxyz * qxyz, axis=-1, keepdims=True) + 1e-10)

        # Small angle approximation
        small_angle = norm_qxyz < 1e-8

        # Normal case: omega = 2 * atan2(||qxyz||, qw) * qxyz / ||qxyz||
        scale = mx.where(
            small_angle,
            2.0 / qw,  # First-order approximation
            2.0 * mx.arctan2(norm_qxyz, qw) / norm_qxyz,
        )
        return scale * qxyz

    def inv(self) -> SO3:
        """Inverse rotation (conjugate for unit quaternion).

        Returns:
            Inverse SO3 element
        """
        qxyz = -self.data[..., :3]
        qw = self.data[..., 3:4]
        return SO3(mx.concatenate([qxyz, qw], axis=-1))

    def __mul__(self, other: SO3) -> SO3:
        """Quaternion multiplication (compose rotations).

        Args:
            other: Another SO3 element

        Returns:
            Composed rotation
        """
        q1 = self.data
        q2 = other.data

        x1, y1, z1, w1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        x2, y2, z2, w2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        return SO3(mx.concatenate([x, y, z, w], axis=-1))

    def act(self, point: mx.array) -> mx.array:
        """Rotate a 3D point.

        Args:
            point: 3D point(s) [..., 3]

        Returns:
            Rotated point(s) [..., 3]
        """
        # q * p * q^-1 where p = [px, py, pz, 0]
        qxyz = self.data[..., :3]
        qw = self.data[..., 3:4]

        # Efficient rotation: p' = p + 2 * qw * (qxyz x p) + 2 * (qxyz x (qxyz x p))
        t = 2.0 * _cross(qxyz, point)
        return point + qw * t + _cross(qxyz, t)

    def matrix(self) -> mx.array:
        """Convert to 3x3 rotation matrix.

        Returns:
            Rotation matrix [..., 3, 3]
        """
        x, y, z, w = (
            self.data[..., 0],
            self.data[..., 1],
            self.data[..., 2],
            self.data[..., 3],
        )

        # Rotation matrix from quaternion
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        batch_shape = self.data.shape[:-1]
        R = mx.zeros(batch_shape + (3, 3), dtype=self.data.dtype)

        R = R.at[..., 0, 0].add(1 - 2 * (yy + zz))
        R = R.at[..., 0, 1].add(2 * (xy - wz))
        R = R.at[..., 0, 2].add(2 * (xz + wy))
        R = R.at[..., 1, 0].add(2 * (xy + wz))
        R = R.at[..., 1, 1].add(1 - 2 * (xx + zz))
        R = R.at[..., 1, 2].add(2 * (yz - wx))
        R = R.at[..., 2, 0].add(2 * (xz - wy))
        R = R.at[..., 2, 1].add(2 * (yz + wx))
        R = R.at[..., 2, 2].add(1 - 2 * (xx + yy))

        return R

    def retr(self, delta: mx.array) -> SO3:
        """Retraction: update rotation by tangent vector.

        Args:
            delta: Tangent vector [..., 3]

        Returns:
            Updated SO3 element
        """
        return self * SO3.exp(delta)

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.data.shape[:-1]

    def __repr__(self) -> str:
        return f"SO3(shape={self.shape}, data={self.data})"
