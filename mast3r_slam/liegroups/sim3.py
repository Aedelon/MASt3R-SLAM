# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Pure PyTorch implementation of Sim3 (Similarity Transform).

Replaces lietorch.Sim3 for CPU/CUDA/MPS compatibility.

Data format: [tx, ty, tz, qx, qy, qz, qw, s] (8 elements)
- Translation: t[3]
- Quaternion: q[4] = [qx, qy, qz, qw] (Hamilton convention, w last)
- Scale: s[1]
"""

from __future__ import annotations

import torch
from torch import Tensor


class Sim3:
    """
    Sim3 similarity transformation (rotation + translation + scale).

    Pure PyTorch implementation compatible with CPU/CUDA/MPS.
    API compatible with lietorch.Sim3.
    """

    embedded_dim: int = 8

    def __init__(self, data: Tensor) -> None:
        """
        Create Sim3 from tensor data.

        Args:
            data: Tensor of shape [..., 8] containing [tx, ty, tz, qx, qy, qz, qw, s]
        """
        self.data = data

    @classmethod
    def Identity(
        cls,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Sim3:
        """
        Create identity Sim3 transformation(s).

        Args:
            batch: Number of identity transforms to create
            device: Target device
            dtype: Data type

        Returns:
            Sim3 with identity transformation(s)
        """
        data = torch.zeros(batch, 1, 8, device=device, dtype=dtype)
        data[..., 6] = 1.0  # qw = 1 (identity quaternion)
        data[..., 7] = 1.0  # s = 1 (identity scale)
        return cls(data)

    def _unpack(self) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack data into translation, quaternion, scale."""
        t = self.data[..., :3]
        q = self.data[..., 3:7]
        s = self.data[..., 7:8]
        return t, q, s

    def _pack(self, t: Tensor, q: Tensor, s: Tensor) -> Tensor:
        """Pack translation, quaternion, scale into data tensor."""
        return torch.cat([t, q, s], dim=-1)

    @staticmethod
    def _quat_normalize(q: Tensor) -> Tensor:
        """Normalize quaternion to unit length."""
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _quat_conj(q: Tensor) -> Tensor:
        """Quaternion conjugate (inverse for unit quaternion)."""
        # q = [qx, qy, qz, qw] -> q* = [-qx, -qy, -qz, qw]
        signs = torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)
        return q * signs

    @staticmethod
    def _quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
        """
        Hamilton quaternion multiplication: q1 * q2.

        Quaternion format: [qx, qy, qz, qw]
        """
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def _quat_rotate(q: Tensor, v: Tensor) -> Tensor:
        """
        Rotate vector v by quaternion q.

        Uses the formula: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        This avoids forming the full rotation matrix.

        Handles broadcasting for various input shapes.
        """
        # Make q and v have the same number of dimensions
        q_ndim = q.dim()
        v_ndim = v.dim()

        if v_ndim > q_ndim:
            # v has more dims: add dimensions to q
            for _ in range(v_ndim - q_ndim):
                q = q.unsqueeze(-2)
        elif q_ndim > v_ndim:
            # q has more dims: add dimensions to v
            for _ in range(q_ndim - v_ndim):
                v = v.unsqueeze(0)

        qxyz = q[..., :3]
        qw = q[..., 3:4]

        # uv = 2 * cross(qxyz, v)
        uv = 2.0 * torch.cross(qxyz, v, dim=-1)

        # result = v + w*uv + cross(qxyz, uv)
        return v + qw * uv + torch.cross(qxyz, uv, dim=-1)

    @staticmethod
    def _quat_to_matrix(q: Tensor) -> Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        x, y, z, w = q.unbind(-1)

        # Pre-compute products
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        # Build rotation matrix
        R = torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],
            dim=-1,
        )

        return R.view(*q.shape[:-1], 3, 3)

    def act(self, points: Tensor) -> Tensor:
        """
        Transform points by this Sim3 transformation.

        Formula: p' = s * R @ p + t

        Args:
            points: Points to transform [..., 3] or [..., N, 3]

        Returns:
            Transformed points with same shape as input
        """
        t, q, s = self._unpack()

        # Rotate (handles broadcasting internally)
        rotated = self._quat_rotate(q, points)

        # Handle broadcasting for s and t to match rotated output shape
        t_ndim = t.dim()
        r_ndim = rotated.dim()

        if r_ndim > t_ndim:
            for _ in range(r_ndim - t_ndim):
                t = t.unsqueeze(-2)
                s = s.unsqueeze(-2)

        # Scale and translate
        return s * rotated + t

    def inv(self) -> Sim3:
        """
        Compute inverse transformation.

        For T = (R, t, s), T^-1 = (R^-1, -s^-1 * R^-1 @ t, s^-1)
        """
        t, q, s = self._unpack()

        s_inv = 1.0 / s
        q_inv = self._quat_conj(q)
        t_inv = -s_inv * self._quat_rotate(q_inv, t)

        return Sim3(self._pack(t_inv, q_inv, s_inv))

    def matrix(self) -> Tensor:
        """
        Convert to 4x4 homogeneous transformation matrix.

        Returns:
            [..., 4, 4] transformation matrix where:
            M[:3, :3] = s * R
            M[:3, 3] = t
            M[3, :] = [0, 0, 0, 1]
        """
        t, q, s = self._unpack()
        R = self._quat_to_matrix(q)

        # Build 4x4 matrix
        batch_shape = self.data.shape[:-1]
        M = torch.zeros(
            *batch_shape, 4, 4, device=self.data.device, dtype=self.data.dtype
        )
        M[..., :3, :3] = s[..., None] * R
        M[..., :3, 3] = t
        M[..., 3, 3] = 1.0

        return M

    def __mul__(self, other: Sim3) -> Sim3:
        """
        Compose two Sim3 transformations: self * other.

        For T1 = (R1, t1, s1) and T2 = (R2, t2, s2):
        T1 * T2 = (R1 @ R2, s1 * R1 @ t2 + t1, s1 * s2)
        """
        t1, q1, s1 = self._unpack()
        t2, q2, s2 = other._unpack()

        # Compose rotation: q_new = q1 * q2
        q_new = self._quat_mul(q1, q2)
        q_new = self._quat_normalize(q_new)

        # Compose translation: t_new = s1 * R1 @ t2 + t1
        t_new = s1 * self._quat_rotate(q1, t2) + t1

        # Compose scale: s_new = s1 * s2
        s_new = s1 * s2

        return Sim3(self._pack(t_new, q_new, s_new))

    def __matmul__(self, other: Sim3) -> Sim3:
        """Alias for __mul__ to support @ operator."""
        return self.__mul__(other)

    @staticmethod
    def _exp_so3(phi: Tensor) -> Tensor:
        """
        SO3 exponential map: axis-angle to quaternion.

        Args:
            phi: Axis-angle rotation vector [..., 3]

        Returns:
            Unit quaternion [..., 4]
        """
        theta_sq = (phi * phi).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-12)

        # Taylor expansion for small angles
        small_angle = theta_sq < 1e-8
        half_theta = 0.5 * theta

        # sin(theta/2) / theta
        imag_factor = torch.where(
            small_angle,
            0.5 - theta_sq / 48.0,  # Taylor: 1/2 - theta^2/48
            torch.sin(half_theta) / theta,
        )

        # cos(theta/2)
        real_factor = torch.where(
            small_angle,
            1.0 - theta_sq / 8.0,  # Taylor: 1 - theta^2/8
            torch.cos(half_theta),
        )

        qxyz = imag_factor * phi
        qw = real_factor

        return torch.cat([qxyz, qw], dim=-1)

    @staticmethod
    def _exp_sim3(xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sim3 exponential map.

        Args:
            xi: Tangent vector [..., 7] = [tau(3), phi(3), sigma(1)]
                - tau: translational velocity
                - phi: rotational velocity (axis-angle)
                - sigma: scale velocity

        Returns:
            (t, q, s): translation, quaternion, scale
        """
        tau = xi[..., :3]  # Translation tangent
        phi = xi[..., 3:6]  # Rotation tangent (axis-angle)
        sigma = xi[..., 6:7]  # Scale tangent

        # Scale: s = exp(sigma)
        s = torch.exp(sigma)

        # Rotation: q = exp(phi)
        q = Sim3._exp_so3(phi)

        # Translation: t = W @ tau (complex formula)
        theta_sq = (phi * phi).sum(dim=-1, keepdim=True)
        theta = torch.sqrt(theta_sq + 1e-12)
        sigma_sq = sigma * sigma

        # Coefficients A, B, C for W matrix
        # W = C*I + A*[phi]_x + B*[phi]_x^2
        EPS = 1e-6

        small_sigma = torch.abs(sigma) < EPS
        small_theta = theta_sq < EPS

        # Case 1: small sigma
        C_small_sigma = torch.ones_like(sigma)

        # Case 2: normal sigma
        C_normal = (s - 1.0) / sigma

        C = torch.where(small_sigma, C_small_sigma, C_normal)

        # A and B coefficients (more complex)
        # Case: small sigma, small theta
        A_ss_st = 0.5 * torch.ones_like(sigma)
        B_ss_st = (1.0 / 6.0) * torch.ones_like(sigma)

        # Case: small sigma, normal theta
        A_ss_nt = (1.0 - torch.cos(theta)) / theta_sq
        B_ss_nt = (theta - torch.sin(theta)) / (theta_sq * theta)

        # Case: normal sigma, small theta
        A_ns_st = ((sigma - 1.0) * s + 1.0) / sigma_sq
        B_ns_st = (s * 0.5 * sigma_sq + s - 1.0 - sigma * s) / (sigma_sq * sigma)

        # Case: normal sigma, normal theta
        a = s * torch.sin(theta)
        b = s * torch.cos(theta)
        c = theta_sq + sigma_sq
        A_ns_nt = (a * sigma + (1.0 - b) * theta) / (theta * c)
        B_ns_nt = (C_normal - ((b - 1.0) * sigma + a * theta) / c) / theta_sq

        # Select A based on cases
        A = torch.where(
            small_sigma,
            torch.where(small_theta, A_ss_st, A_ss_nt),
            torch.where(small_theta, A_ns_st, A_ns_nt),
        )

        B = torch.where(
            small_sigma,
            torch.where(small_theta, B_ss_st, B_ss_nt),
            torch.where(small_theta, B_ns_st, B_ns_nt),
        )

        # t = W @ tau = C*tau + A*(phi x tau) + B*(phi x (phi x tau))
        t = C * tau

        # phi x tau
        cross1 = torch.cross(phi, tau, dim=-1)
        t = t + A * cross1

        # phi x (phi x tau)
        cross2 = torch.cross(phi, cross1, dim=-1)
        t = t + B * cross2

        return t, q, s

    def retr(self, xi: Tensor) -> Sim3:
        """
        Retraction: apply tangent vector update.

        Computes: exp(xi) * self

        Args:
            xi: Tangent vector [..., 7] = [tau(3), phi(3), sigma(1)]

        Returns:
            Updated Sim3 transformation
        """
        t, q, s = self._unpack()

        # Compute delta transformation
        dt, dq, ds = self._exp_sim3(xi)

        # Compose: delta * self
        # q_new = dq * q
        q_new = self._quat_mul(dq, q)
        q_new = self._quat_normalize(q_new)

        # t_new = ds * dR @ t + dt
        t_new = ds * self._quat_rotate(dq, t) + dt

        # s_new = ds * s
        s_new = ds * s

        return Sim3(self._pack(t_new, q_new, s_new))

    def __repr__(self) -> str:
        return f"Sim3(data={self.data.shape})"
