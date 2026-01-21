# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""MPSGraph kernel implementations for SLAM operations.

Uses Metal Performance Shaders Graph for GPU-accelerated operations on Apple Silicon.
"""

from __future__ import annotations

import numpy as np

# Check if PyObjC and Metal frameworks are available
_MPSGRAPH_AVAILABLE = False
_mps = None
_mtl = None

try:
    import Metal as mtl
    import MetalPerformanceShadersGraph as mps

    _mtl = mtl
    _mps = mps
    _MPSGRAPH_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if MPSGraph backend is available."""
    return _MPSGRAPH_AVAILABLE


def _get_device():
    """Get default Metal device."""
    if not _MPSGRAPH_AVAILABLE:
        raise RuntimeError(
            "MPSGraph not available. Install PyObjC: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShadersGraph"
        )
    return _mtl.MTLCreateSystemDefaultDevice()


def _create_graph():
    """Create a new MPSGraph instance."""
    return _mps.MPSGraph.alloc().init()


def _numpy_to_mps_tensor(graph, data: np.ndarray, name: str = None):
    """Convert numpy array to MPSGraph placeholder."""
    shape = [int(s) for s in data.shape]

    # Map numpy dtype to MPS data type
    dtype_map = {
        np.float32: _mps.MPSDataTypeFloat32,
        np.float16: _mps.MPSDataTypeFloat16,
        np.int32: _mps.MPSDataTypeInt32,
        np.int64: _mps.MPSDataTypeInt64,
        np.bool_: _mps.MPSDataTypeBool,
    }

    mps_dtype = dtype_map.get(data.dtype.type, _mps.MPSDataTypeFloat32)

    return graph.placeholderWithShape_dataType_name_(shape, mps_dtype, name)


def _run_graph(graph, feeds: dict, targets: list, device=None):
    """Execute MPSGraph with given feeds and return results."""
    if device is None:
        device = _get_device()

    # Create command queue
    queue = device.newCommandQueue()

    # Create tensor data for feeds
    feed_dict = {}
    for placeholder, data in feeds.items():
        # Ensure contiguous array
        data = np.ascontiguousarray(data)
        tensor_data = _mps.MPSGraphTensorData.alloc().initWithDevice_data_shape_dataType_(
            device,
            data.tobytes(),
            [int(s) for s in data.shape],
            _mps.MPSDataTypeFloat32 if data.dtype == np.float32 else _mps.MPSDataTypeFloat16,
        )
        feed_dict[placeholder] = tensor_data

    # Run graph
    results = graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations_(
        queue, feed_dict, targets, None
    )

    # Convert results back to numpy
    outputs = []
    for target in targets:
        tensor_data = results[target]
        shape = [int(s) for s in tensor_data.shape()]
        dtype = np.float32  # Default
        data = np.frombuffer(tensor_data.mpsndarray().bytes(), dtype=dtype).reshape(shape)
        outputs.append(data.copy())

    return outputs


# =============================================================================
# Iterative Projection Kernel
# =============================================================================


def iter_proj(
    rays_with_grad: np.ndarray,
    pts3d_norm: np.ndarray,
    p_init: np.ndarray,
    max_iter: int = 10,
    lambda_init: float = 1e-8,
    convergence_thresh: float = 1e-6,
    use_metal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative projection matching using Metal GPU acceleration.

    Projects 3D points onto ray image using Levenberg-Marquardt optimization.

    Args:
        rays_with_grad: Ray image with gradients [B, H, W, 9]
        pts3d_norm: Normalized 3D points [B, N, 3]
        p_init: Initial pixel positions [B, N, 2]
        max_iter: Maximum LM iterations
        lambda_init: Initial damping factor
        convergence_thresh: Convergence threshold
        use_metal: Use Metal GPU acceleration (default True)

    Returns:
        p_final: Final pixel positions [B, N, 2]
        valid: Validity mask [B, N]
    """
    # Try Metal GPU acceleration
    if use_metal and _MPSGRAPH_AVAILABLE:
        try:
            from .metal_runner import get_runner
            runner = get_runner()
            return runner.iter_proj(
                rays_with_grad, pts3d_norm, p_init,
                max_iter, lambda_init, convergence_thresh
            )
        except Exception:
            pass  # Fall back to numpy

    # Fallback to numpy implementation
    return _iter_proj_numpy(
        rays_with_grad, pts3d_norm, p_init, max_iter, lambda_init, convergence_thresh
    )


def _iter_proj_numpy(
    rays_with_grad: np.ndarray,
    pts3d_norm: np.ndarray,
    p_init: np.ndarray,
    max_iter: int,
    lambda_init: float,
    convergence_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy fallback for iter_proj - optimized version."""
    b, h, w, _ = rays_with_grad.shape
    n = pts3d_norm.shape[1]

    p = p_init.astype(np.float32).copy()
    lam = lambda_init

    for _ in range(max_iter):
        # Clamp to valid range
        p_clamped = np.stack(
            [
                np.clip(p[..., 0], 0, w - 1.001),
                np.clip(p[..., 1], 0, h - 1.001),
            ],
            axis=-1,
        )

        # Bilinear interpolation
        x0 = np.floor(p_clamped[..., 0]).astype(np.int32)
        y0 = np.floor(p_clamped[..., 1]).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = (p_clamped[..., 0] - x0)[..., None]
        fy = (p_clamped[..., 1] - y0)[..., None]

        # Sample 4 corners - vectorized for all batches
        rays_sampled = np.zeros((b, n, 9), dtype=np.float32)
        for bi in range(b):
            v00 = rays_with_grad[bi, y0[bi], x0[bi], :]
            v01 = rays_with_grad[bi, y1[bi], x0[bi], :]
            v10 = rays_with_grad[bi, y0[bi], x1[bi], :]
            v11 = rays_with_grad[bi, y1[bi], x1[bi], :]

            rays_sampled[bi] = (
                (1 - fx[bi]) * (1 - fy[bi]) * v00
                + (1 - fx[bi]) * fy[bi] * v01
                + fx[bi] * (1 - fy[bi]) * v10
                + fx[bi] * fy[bi] * v11
            )

        # Extract rays and gradients
        rays = rays_sampled[..., :3]
        grad_x = rays_sampled[..., 3:6]
        grad_y = rays_sampled[..., 6:9]

        # Residual
        r = rays - pts3d_norm

        # Jacobian [B, N, 3, 2]
        J = np.stack([grad_x, grad_y], axis=-1)

        # JtJ [B, N, 2, 2]
        JtJ = np.einsum("bnki,bnkj->bnij", J, J)

        # Add damping
        JtJ[..., 0, 0] += lam
        JtJ[..., 1, 1] += lam

        # Jtr [B, N, 2]
        Jtr = np.einsum("bnki,bnk->bni", J, r)

        # Solve 2x2 system analytically
        a = JtJ[..., 0, 0]
        b_val = JtJ[..., 0, 1]
        c = JtJ[..., 1, 0]
        d = JtJ[..., 1, 1]

        det = a * d - b_val * c
        det = np.maximum(det, 1e-10)
        inv_det = 1.0 / det

        delta_x = -(d * Jtr[..., 0] - b_val * Jtr[..., 1]) * inv_det
        delta_y = -(-c * Jtr[..., 0] + a * Jtr[..., 1]) * inv_det
        delta = np.stack([delta_x, delta_y], axis=-1)

        # Update
        p = p + delta

        # Check convergence
        delta_norm = np.sqrt(np.sum(delta * delta, axis=-1))
        if np.max(delta_norm) < convergence_thresh:
            break

    # Final clamp and validity
    p_final = np.stack(
        [
            np.clip(p[..., 0], 0, w - 1),
            np.clip(p[..., 1], 0, h - 1),
        ],
        axis=-1,
    )

    valid = (p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h)

    return p_final, valid


# =============================================================================
# Gauss-Newton Ray Alignment
# =============================================================================


def gauss_newton_rays(
    Twc: np.ndarray,
    Xs: np.ndarray,
    Cs: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    idx_ii2jj: np.ndarray,
    valid_match: np.ndarray,
    Q: np.ndarray,
    sigma_ray: float = 0.003,
    sigma_dist: float = 10.0,
    C_thresh: float = 0.0,
    Q_thresh: float = 1.5,
    max_iter: int = 10,
    delta_thresh: float = 1e-4,
    pin: int = 1,
    use_metal: bool = True,
) -> np.ndarray:
    """Gauss-Newton optimization for ray alignment.

    Args:
        Twc: Poses [num_kf, 8] as (tx, ty, tz, qx, qy, qz, qw, scale)
        Xs: 3D points [num_kf, num_pts, 3]
        Cs: Confidences [num_kf, num_pts, 1]
        ii: Source keyframe indices [num_edges]
        jj: Target keyframe indices [num_edges]
        idx_ii2jj: Point correspondences [num_edges, num_pts]
        valid_match: Match validity [num_edges, num_pts, 1]
        Q: Match confidence [num_edges, num_pts, 1]
        sigma_ray: Ray residual weight
        sigma_dist: Distance residual weight
        C_thresh: Confidence threshold
        Q_thresh: Match quality threshold
        max_iter: Maximum iterations
        delta_thresh: Convergence threshold
        pin: Number of poses to fix (usually 1)
        use_metal: Use Metal GPU acceleration (default True)

    Returns:
        Updated Twc poses
    """
    # Try Metal GPU acceleration
    if use_metal and _MPSGRAPH_AVAILABLE:
        try:
            from .gn_metal_runner import get_gn_runner
            runner = get_gn_runner()
            return runner.gauss_newton_rays(
                Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
                sigma_ray, sigma_dist, C_thresh, Q_thresh,
                max_iter, delta_thresh, pin,
            )
        except Exception:
            pass  # Fall back to numpy

    # Fallback to numpy implementation
    from .gauss_newton import gauss_newton_rays as gn_impl
    return gn_impl(
        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
        sigma_ray, sigma_dist, C_thresh, Q_thresh,
        max_iter, delta_thresh, pin,
    )


def gauss_newton_calib(
    Twc: np.ndarray,
    Xs: np.ndarray,
    Cs: np.ndarray,
    K: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    idx_ii2jj: np.ndarray,
    valid_match: np.ndarray,
    Q: np.ndarray,
    img_size: tuple[int, int],
    pixel_border: int = 0,
    z_eps: float = 0.0,
    sigma_pixel: float = 1.0,
    sigma_depth: float = 0.1,
    C_thresh: float = 0.0,
    Q_thresh: float = 1.5,
    max_iter: int = 10,
    delta_thresh: float = 1e-4,
    pin: int = 1,
    use_metal: bool = True,
) -> np.ndarray:
    """Gauss-Newton optimization for calibrated projection residuals.

    Args:
        Twc: Poses [num_kf, 8] as (tx, ty, tz, qx, qy, qz, qw, scale)
        Xs: 3D points [num_kf, num_pts, 3]
        Cs: Confidences [num_kf, num_pts]
        K: Intrinsic matrix [3, 3] or [4] = (fx, fy, cx, cy)
        ii: Source keyframe indices [num_edges]
        jj: Target keyframe indices [num_edges]
        idx_ii2jj: Point correspondences [num_edges, num_pts]
        valid_match: Match validity [num_edges, num_pts]
        Q: Match confidence [num_edges, num_pts]
        img_size: (width, height)
        pixel_border: Border to exclude
        z_eps: Minimum depth
        sigma_pixel: Pixel residual weight
        sigma_depth: Depth residual weight
        C_thresh: Confidence threshold
        Q_thresh: Match quality threshold
        max_iter: Maximum iterations
        delta_thresh: Convergence threshold
        pin: Number of poses to fix
        use_metal: Use Metal GPU acceleration

    Returns:
        Updated Twc poses
    """
    # Try Metal GPU acceleration
    if use_metal and _MPSGRAPH_AVAILABLE:
        try:
            from .gn_calib_metal_runner import get_gn_calib_runner
            runner = get_gn_calib_runner()
            return runner.gauss_newton_calib(
                Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
                img_size, pixel_border, z_eps, sigma_pixel, sigma_depth,
                C_thresh, Q_thresh, max_iter, delta_thresh, pin,
            )
        except Exception:
            pass  # Fall back to numpy

    # Fallback to numpy implementation
    from .gauss_newton_calib import gauss_newton_calib as gn_calib_impl
    return gn_calib_impl(
        Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
        img_size, pixel_border, z_eps, sigma_pixel, sigma_depth,
        C_thresh, Q_thresh, max_iter, delta_thresh, pin,
    )


def gauss_newton_points(
    Twc: np.ndarray,
    Xs: np.ndarray,
    Cs: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    idx_ii2jj: np.ndarray,
    valid_match: np.ndarray,
    Q: np.ndarray,
    sigma_point: float = 0.01,
    C_thresh: float = 0.0,
    Q_thresh: float = 1.5,
    max_iter: int = 10,
    delta_thresh: float = 1e-4,
    pin: int = 1,
    use_metal: bool = True,
) -> np.ndarray:
    """Gauss-Newton optimization for 3D point alignment.

    This is the point-based mode (vs ray-based). Uses direct 3D point
    error without ray normalization.

    Args:
        Twc: Poses [num_kf, 8] as (tx, ty, tz, qx, qy, qz, qw, scale)
        Xs: 3D points [num_kf, num_pts, 3]
        Cs: Confidences [num_kf, num_pts]
        ii: Source keyframe indices [num_edges]
        jj: Target keyframe indices [num_edges]
        idx_ii2jj: Point correspondences [num_edges, num_pts]
        valid_match: Match validity [num_edges, num_pts]
        Q: Match confidence [num_edges, num_pts]
        sigma_point: Point residual weight
        C_thresh: Confidence threshold
        Q_thresh: Match quality threshold
        max_iter: Maximum iterations
        delta_thresh: Convergence threshold
        pin: Number of poses to fix
        use_metal: Use Metal GPU acceleration

    Returns:
        Updated Twc poses
    """
    # Try Metal GPU acceleration
    if use_metal and _MPSGRAPH_AVAILABLE:
        try:
            from .gn_points_metal_runner import get_gn_points_runner
            runner = get_gn_points_runner()
            return runner.gauss_newton_points(
                Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
                sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh, pin,
            )
        except Exception:
            pass  # Fall back to numpy

    # Fallback to numpy implementation
    from .gauss_newton_points import gauss_newton_points as gn_points_impl
    return gn_points_impl(
        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
        sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh, pin,
    )


# =============================================================================
# Match Refinement
# =============================================================================


def refine_matches(
    D11: np.ndarray,
    D21: np.ndarray,
    p1: np.ndarray,
    radius: int = 3,
    dilation_max: int = 0,
    use_metal: bool = True,
) -> np.ndarray:
    """Refine matches using local descriptor search.

    Args:
        D11: Descriptors from view 1 [B, H, W, D]
        D21: Descriptors from view 2 [B, N, D]
        p1: Current pixel positions [B, N, 2]
        radius: Search radius
        dilation_max: Maximum dilation for multi-scale search
        use_metal: Use Metal GPU acceleration (default True)

    Returns:
        Refined pixel positions [B, N, 2]
    """
    # Try Metal GPU acceleration
    if use_metal and _MPSGRAPH_AVAILABLE:
        try:
            from .refine_metal_runner import get_refine_runner
            runner = get_refine_runner()
            return runner.refine_matches(D11, D21, p1, radius, dilation_max)
        except Exception:
            pass  # Fall back to numpy

    return _refine_matches_numpy(D11, D21, p1, radius, dilation_max)


def _refine_matches_numpy(
    D11: np.ndarray,
    D21: np.ndarray,
    p1: np.ndarray,
    radius: int,
    dilation_max: int,
) -> np.ndarray:
    """Numpy implementation of match refinement."""
    b, h, w, d = D11.shape
    n = p1.shape[1]

    p_refined = p1.astype(np.int32).copy()

    # For each dilation level
    dilations = list(range(max(1, dilation_max), 0, -1))

    for dilation in dilations:
        for bi in range(b):
            for ni in range(n):
                cx, cy = int(p1[bi, ni, 0]), int(p1[bi, ni, 1])
                query_desc = D21[bi, ni, :]

                best_score = -np.inf
                best_x, best_y = cx, cy

                # Search in window
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny = cy + dy * dilation
                        nx = cx + dx * dilation

                        if 0 <= nx < w and 0 <= ny < h:
                            ref_desc = D11[bi, ny, nx, :]
                            score = np.dot(query_desc, ref_desc)
                            if score > best_score:
                                best_score = score
                                best_x, best_y = nx, ny

                p_refined[bi, ni, 0] = best_x
                p_refined[bi, ni, 1] = best_y

    return p_refined
