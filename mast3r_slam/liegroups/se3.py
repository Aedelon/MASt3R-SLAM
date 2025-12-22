# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Pure PyTorch implementation of SE3 (Rigid Body Transform).

Used for trajectory evaluation and visualization.

Data format: [tx, ty, tz, qx, qy, qz, qw] (7 elements)
- Translation: t[3]
- Quaternion: q[4] = [qx, qy, qz, qw] (Hamilton convention, w last)
"""

from __future__ import annotations

import torch
from torch import Tensor


class SE3:
    """
    SE3 rigid body transformation (rotation + translation).

    Pure PyTorch implementation compatible with CPU/CUDA/MPS.
    Used primarily for trajectory export (no scale).
    """

    embedded_dim: int = 7

    def __init__(self, data: Tensor) -> None:
        """
        Create SE3 from tensor data.

        Args:
            data: Tensor of shape [..., 7] containing [tx, ty, tz, qx, qy, qz, qw]
        """
        self.data = data

    @classmethod
    def Identity(
        cls,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SE3:
        """Create identity SE3 transformation(s)."""
        data = torch.zeros(batch, 1, 7, device=device, dtype=dtype)
        data[..., 6] = 1.0  # qw = 1 (identity quaternion)
        return cls(data)

    def _unpack(self) -> tuple[Tensor, Tensor]:
        """Unpack data into translation, quaternion."""
        t = self.data[..., :3]
        q = self.data[..., 3:7]
        return t, q

    @staticmethod
    def _quat_to_matrix(q: Tensor) -> Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        x, y, z, w = q.unbind(-1)

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

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

    def matrix(self) -> Tensor:
        """
        Convert to 4x4 homogeneous transformation matrix.

        Returns:
            [..., 4, 4] transformation matrix
        """
        t, q = self._unpack()
        R = self._quat_to_matrix(q)

        batch_shape = self.data.shape[:-1]
        M = torch.zeros(
            *batch_shape, 4, 4, device=self.data.device, dtype=self.data.dtype
        )
        M[..., :3, :3] = R
        M[..., :3, 3] = t
        M[..., 3, 3] = 1.0

        return M

    def __repr__(self) -> str:
        return f"SE3(data={self.data.shape})"
