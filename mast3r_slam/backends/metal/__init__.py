# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Metal backend for Apple Silicon GPU.

Uses native Metal compute shaders for high performance on M1/M2/M3 chips.
Supports MPS tensors directly for zero-copy GPU operations.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from mast3r_slam.backends.base import BackendBase


def _prepare_tensor(t: Tensor) -> Tensor:
    """Prepare tensor for Metal kernel - supports MPS zero-copy."""
    # MPS tensors can be used directly (zero-copy via shared memory)
    # CPU tensors will be copied to Metal buffers
    return t.contiguous()


class MetalBackend(BackendBase):
    """Metal backend using Apple Silicon GPU with MPS zero-copy support."""

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
        # Metal supports MPS tensors directly (zero-copy)
        result = self._backend.iter_proj(
            _prepare_tensor(rays_img_with_grad),
            _prepare_tensor(pts_3d_norm),
            _prepare_tensor(p_init),
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
        result = self._backend.refine_matches(
            _prepare_tensor(D11),
            _prepare_tensor(D21),
            _prepare_tensor(p1),
            radius,
            dilation_max,
        )
        return (result[0],)

    def iter_proj_and_refine(
        self,
        rays_img_with_grad: Tensor,
        pts_3d_norm: Tensor,
        p_init: Tensor,
        max_iter: int,
        lambda_init: float,
        cost_thresh: float,
        D11: Tensor,
        D21: Tensor,
        radius: int,
        dilation_max: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Batched iter_proj + refine_matches in single command buffer."""
        result = self._backend.iter_proj_and_refine_batched(
            _prepare_tensor(rays_img_with_grad),
            _prepare_tensor(pts_3d_norm),
            _prepare_tensor(p_init),
            max_iter,
            lambda_init,
            cost_thresh,
            _prepare_tensor(D11),
            _prepare_tensor(D21),
            radius,
            dilation_max,
        )
        return result[0], result[1], result[2]

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
        """
        Gauss-Newton optimization with ray alignment residuals.

        Uses Metal GPU for Jacobian computation + PyTorch MPS for linear solve.
        Supports MPS tensors directly for zero-copy operations.
        """
        device = poses.device
        result = self._backend.gauss_newton_rays(
            _prepare_tensor(poses),
            _prepare_tensor(points),
            _prepare_tensor(confidences),
            _prepare_tensor(ii),
            _prepare_tensor(jj),
            _prepare_tensor(idx_ii2jj),
            _prepare_tensor(valid_match),
            _prepare_tensor(Q),
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        # Return dx on original device
        return [r.to(device) for r in result]

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
        """
        Gauss-Newton optimization with calibrated projection residuals.

        Uses Metal GPU for Jacobian computation + PyTorch MPS for linear solve.
        Supports MPS tensors directly for zero-copy operations.
        """
        device = poses.device
        result = self._backend.gauss_newton_calib(
            _prepare_tensor(poses),
            _prepare_tensor(points),
            _prepare_tensor(confidences),
            _prepare_tensor(K),
            _prepare_tensor(ii),
            _prepare_tensor(jj),
            _prepare_tensor(idx_ii2jj),
            _prepare_tensor(valid_match),
            _prepare_tensor(Q),
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
        return [r.to(device) for r in result]
