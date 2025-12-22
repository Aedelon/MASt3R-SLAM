# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Metal backend for Apple Silicon GPU.

Uses native Metal compute shaders for high performance on M1/M2/M3 chips.
"""

from __future__ import annotations

from typing import List, Tuple

from torch import Tensor

from mast3r_slam.backends.base import BackendBase


class MetalBackend(BackendBase):
    """Metal backend using Apple Silicon GPU."""

    def __init__(self) -> None:
        # Import here to fail gracefully if not built
        import mast3r_slam_metal_backends

        self._backend = mast3r_slam_metal_backends

        # Initialize Metal device
        if not self._backend.initialize():
            raise RuntimeError("Failed to initialize Metal backend")

    @property
    def name(self) -> str:
        return "metal"

    def iter_proj(
        self,
        rays_img_with_grad: Tensor,
        pts_3d_norm: Tensor,
        p_init: Tensor,
        max_iter: int,
        lambda_init: float,
        cost_thresh: float,
    ) -> Tuple[Tensor, Tensor]:
        # Metal kernels work on CPU tensors, copy data
        result = self._backend.iter_proj(
            rays_img_with_grad.cpu().contiguous(),
            pts_3d_norm.cpu().contiguous(),
            p_init.cpu().contiguous(),
            max_iter,
            lambda_init,
            cost_thresh,
        )
        # Move results back to MPS if needed
        device = p_init.device
        return result[0].to(device), result[1].to(device)

    def refine_matches(
        self,
        D11: Tensor,
        D21: Tensor,
        p1: Tensor,
        radius: int,
        dilation_max: int,
    ) -> Tuple[Tensor]:
        result = self._backend.refine_matches(
            D11.cpu().contiguous(),
            D21.cpu().contiguous(),
            p1.cpu().contiguous(),
            radius,
            dilation_max,
        )
        device = p1.device
        return (result[0].to(device),)

    def gauss_newton_rays(
        self,
        poses: Tensor,
        points: Tensor,
        confidences: Tensor,
        ii: Tensor,
        jj: Tensor,
        idx_ii2jj: Tensor,
        valid_match: Tensor,
        Q: Tensor,
        sigma_ray: float,
        sigma_dist: float,
        C_thresh: float,
        Q_thresh: float,
        max_iter: int,
        delta_thresh: float,
    ) -> List[Tensor]:
        return self._backend.gauss_newton_rays(
            poses.cpu().contiguous(),
            points.cpu().contiguous(),
            confidences.cpu().contiguous(),
            ii.cpu().contiguous(),
            jj.cpu().contiguous(),
            idx_ii2jj.cpu().contiguous(),
            valid_match.cpu().contiguous(),
            Q.cpu().contiguous(),
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

    def gauss_newton_calib(
        self,
        poses: Tensor,
        points: Tensor,
        confidences: Tensor,
        K: Tensor,
        ii: Tensor,
        jj: Tensor,
        idx_ii2jj: Tensor,
        valid_match: Tensor,
        Q: Tensor,
        height: int,
        width: int,
        pixel_border: int,
        z_eps: float,
        sigma_pixel: float,
        sigma_depth: float,
        C_thresh: float,
        Q_thresh: float,
        max_iter: int,
        delta_thresh: float,
    ) -> List[Tensor]:
        return self._backend.gauss_newton_calib(
            poses.cpu().contiguous(),
            points.cpu().contiguous(),
            confidences.cpu().contiguous(),
            K.cpu().contiguous(),
            ii.cpu().contiguous(),
            jj.cpu().contiguous(),
            idx_ii2jj.cpu().contiguous(),
            valid_match.cpu().contiguous(),
            Q.cpu().contiguous(),
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
