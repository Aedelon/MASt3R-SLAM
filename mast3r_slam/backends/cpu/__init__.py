# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
CPU backend with OpenMP parallelization and SIMD optimizations.

Provides portable fallback when CUDA is not available.
Uses AVX2 on x86_64, NEON on ARM64.
"""

from __future__ import annotations

from typing import List, Tuple

from torch import Tensor

from mast3r_slam.backends.base import BackendBase


class CPUBackend(BackendBase):
    """CPU backend using OpenMP + SIMD extension."""

    def __init__(self) -> None:
        # Import here to fail gracefully if not built
        import mast3r_slam_cpu_backends

        self._backend = mast3r_slam_cpu_backends

    @property
    def name(self) -> str:
        return "cpu"

    def iter_proj(
        self,
        rays_img_with_grad: Tensor,
        pts_3d_norm: Tensor,
        p_init: Tensor,
        max_iter: int,
        lambda_init: float,
        cost_thresh: float,
    ) -> Tuple[Tensor, Tensor]:
        result = self._backend.iter_proj(
            rays_img_with_grad.contiguous(),
            pts_3d_norm.contiguous(),
            p_init.contiguous(),
            max_iter,
            lambda_init,
            cost_thresh,
        )
        return result[0], result[1]

    def refine_matches(
        self,
        D11: Tensor,
        D21: Tensor,
        p1: Tensor,
        radius: int,
        dilation_max: int,
    ) -> Tuple[Tensor]:
        return tuple(
            self._backend.refine_matches(
                D11.contiguous(),
                D21.contiguous(),
                p1.contiguous(),
                radius,
                dilation_max,
            )
        )

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
            poses.contiguous(),
            points.contiguous(),
            confidences.contiguous(),
            ii.contiguous(),
            jj.contiguous(),
            idx_ii2jj.contiguous(),
            valid_match.contiguous(),
            Q.contiguous(),
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
            poses.contiguous(),
            points.contiguous(),
            confidences.contiguous(),
            K.contiguous(),
            ii.contiguous(),
            jj.contiguous(),
            idx_ii2jj.contiguous(),
            valid_match.contiguous(),
            Q.contiguous(),
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
