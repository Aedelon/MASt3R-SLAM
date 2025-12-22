# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Abstract backend interface for multi-platform compute kernels.

Supports CUDA, CPU (OpenMP + SIMD), and future Metal backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import Tensor


class BackendBase(ABC):
    """
    Abstract interface for compute backends.

    All backends must implement these methods for:
    - Iterative projection matching
    - Match refinement with descriptors
    - Gauss-Newton optimization (ray-based and calibrated)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        ...

    @abstractmethod
    def iter_proj(
        self,
        rays_img_with_grad: Tensor,
        pts_3d_norm: Tensor,
        p_init: Tensor,
        max_iter: int,
        lambda_init: float,
        cost_thresh: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Iterative projection for point matching.

        Uses Levenberg-Marquardt optimization to find pixel locations
        where 3D point rays match the ray image.

        Args:
            rays_img_with_grad: Ray image with gradients [B, H, W, 9]
                                (rays[3] + grad_x[3] + grad_y[3])
            pts_3d_norm: Normalized 3D points to project [B, N, 3]
            p_init: Initial pixel guesses [B, N, 2]
            max_iter: Maximum LM iterations
            lambda_init: Initial LM damping factor
            cost_thresh: Convergence threshold

        Returns:
            (p_new, converged): Updated pixel positions [B, N, 2] and
                                convergence flags [B, N]
        """
        ...

    @abstractmethod
    def refine_matches(
        self,
        D11: Tensor,
        D21: Tensor,
        p1: Tensor,
        radius: int,
        dilation_max: int,
    ) -> Tuple[Tensor]:
        """
        Refine match positions using descriptor correlation.

        Searches in a local neighborhood for best descriptor match.

        Args:
            D11: Descriptor image [B, H, W, F] (half precision)
            D21: Flattened descriptors [B, N, F] (half precision)
            p1: Current pixel positions [B, N, 2] (long)
            radius: Search radius
            dilation_max: Maximum dilation factor

        Returns:
            (p1_new,): Refined pixel positions [B, N, 2]
        """
        ...

    @abstractmethod
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
        Gauss-Newton optimization for ray-based alignment.

        Optimizes Sim3 poses to minimize ray alignment error.
        Modifies poses tensor in-place.

        Args:
            poses: Sim3 pose data [N_poses, 8], modified in-place
            points: 3D point clouds [N_poses, H*W, 3]
            confidences: Point confidences [N_poses, H*W, 1]
            ii: Source frame indices [N_edges]
            jj: Target frame indices [N_edges]
            idx_ii2jj: Point correspondences [N_edges, H*W]
            valid_match: Valid match mask [N_edges, H*W]
            Q: Match quality scores [N_edges, H*W, 1]
            sigma_ray: Ray alignment sigma
            sigma_dist: Distance sigma
            C_thresh: Confidence threshold
            Q_thresh: Quality threshold
            max_iter: Maximum GN iterations
            delta_thresh: Convergence threshold

        Returns:
            Empty list (poses modified in-place)
        """
        ...

    @abstractmethod
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
        Gauss-Newton optimization with camera calibration.

        Optimizes Sim3 poses using calibrated reprojection error.
        Modifies poses tensor in-place.

        Args:
            poses: Sim3 pose data [N_poses, 8], modified in-place
            points: 3D point clouds [N_poses, H*W, 3]
            confidences: Point confidences [N_poses, H*W, 1]
            K: Camera intrinsics [3, 3]
            ii: Source frame indices [N_edges]
            jj: Target frame indices [N_edges]
            idx_ii2jj: Point correspondences [N_edges, H*W]
            valid_match: Valid match mask [N_edges, H*W]
            Q: Match quality scores [N_edges, H*W, 1]
            height: Image height
            width: Image width
            pixel_border: Border exclusion zone
            z_eps: Minimum depth threshold
            sigma_pixel: Pixel error sigma
            sigma_depth: Depth error sigma
            C_thresh: Confidence threshold
            Q_thresh: Quality threshold
            max_iter: Maximum GN iterations
            delta_thresh: Convergence threshold

        Returns:
            Empty list (poses modified in-place)
        """
        ...
