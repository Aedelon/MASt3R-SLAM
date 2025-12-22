# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
CUDA backend wrapper.

Wraps the existing mast3r_slam_backends CUDA extension.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from mast3r_slam.backends.base import BackendBase


class CUDABackend(BackendBase):
    """CUDA backend using existing mast3r_slam_backends extension."""

    def __init__(self) -> None:
        # Import here to fail gracefully if CUDA not available
        import mast3r_slam_backends

        self._backend = mast3r_slam_backends

    @property
    def name(self) -> str:
        return "cuda"

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
            rays_img_with_grad,
            pts_3d_norm,
            p_init,
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
        return tuple(self._backend.refine_matches(D11, D21, p1, radius, dilation_max))

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
            poses,
            points,
            confidences,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q,
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
            poses,
            points,
            confidences,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q,
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
