# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Iterative projection matching for MLX-MASt3R-SLAM."""

from __future__ import annotations

import mlx.core as mx

from mlx_mast3r_slam.config import get_config
from mlx_mast3r_slam.image import img_gradient


def match(
    X11: mx.array,
    X21: mx.array,
    D11: mx.array,
    D21: mx.array,
    idx_1_to_2_init: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Match points between two views using iterative projection.

    Args:
        X11: 3D points from view 1 in view 1 frame [B, H, W, 3]
        X21: 3D points from view 2 in view 1 frame [B, H, W, 3]
        D11: Descriptors from view 1 [B, H, W, D]
        D21: Descriptors from view 2 [B, H, W, D]
        idx_1_to_2_init: Initial correspondence indices [B, H*W]

    Returns:
        idx_1_to_2: Correspondence indices [B, H*W]
        valid_match2: Validity mask [B, H*W, 1]
    """
    config = get_config()
    use_simple_match = config.get("matching", {}).get("use_simple", True)

    if use_simple_match:
        return match_simple(X11, X21, D11, D21, idx_1_to_2_init)
    else:
        return match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)


def match_simple(
    X11: mx.array,
    X21: mx.array,
    D11: mx.array,
    D21: mx.array,
    idx_1_to_2_init: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Simple matching using identity correspondence + 3D distance validation.

    This is a memory-efficient fallback when iterative projection is too slow.
    Uses the assumption that MASt3R already aligns points between views.

    Args:
        X11: 3D points from view 1 [B, H, W, 3]
        X21: 3D points from view 2 [B, H, W, 3]
        D11: Descriptors from view 1 [B, H, W, D]
        D21: Descriptors from view 2 [B, H, W, D]
        idx_1_to_2_init: Initial correspondence indices [B, H*W] (used if provided)

    Returns:
        idx_1_to_2: Correspondence indices [B, H*W]
        valid_match2: Validity mask [B, H*W, 1]
    """
    config = get_config()
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    n = h * w

    # Use initial indices if provided, otherwise use identity
    if idx_1_to_2_init is not None:
        idx_1_to_2 = idx_1_to_2_init
    else:
        idx_1_to_2 = mx.broadcast_to(mx.arange(n)[None, :], (b, n))

    # Flatten points
    X11_flat = X11.reshape(b, n, 3)
    X21_flat = X21.reshape(b, n, 3)

    # Sample X11 at correspondence indices
    X11_sampled = mx.zeros((b, n, 3), dtype=X11.dtype)
    for bi in range(b):
        X11_sampled = X11_sampled.at[bi].add(X11_flat[bi, idx_1_to_2[bi], :])

    # Compute 3D distances
    dists = mx.sqrt(mx.sum((X11_sampled - X21_flat) ** 2, axis=-1))

    # Valid if distance is below threshold
    valid = dists < cfg["dist_thresh"]

    return idx_1_to_2, valid[:, :, None]


def pixel_to_lin(p: mx.array, w: int) -> mx.array:
    """Convert 2D pixel coordinates to linear indices.

    Args:
        p: Pixel coordinates [..., 2] with (u, v)
        w: Image width

    Returns:
        Linear indices [...]
    """
    return p[..., 0] + w * p[..., 1]


def lin_to_pixel(idx: mx.array, w: int) -> mx.array:
    """Convert linear indices to 2D pixel coordinates.

    Args:
        idx: Linear indices [...]
        w: Image width

    Returns:
        Pixel coordinates [..., 2] with (u, v)
    """
    u = idx % w
    v = idx // w
    return mx.stack([u, v], axis=-1)


def normalize_rays(X: mx.array) -> mx.array:
    """Normalize 3D points to unit rays.

    Args:
        X: 3D points [..., 3]

    Returns:
        Unit rays [..., 3]
    """
    norm = mx.sqrt(mx.sum(X * X, axis=-1, keepdims=True) + 1e-10)
    return X / norm


def prep_for_iter_proj(
    X11: mx.array,
    X21: mx.array,
    idx_1_to_2_init: mx.array | None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Prepare data for iterative projection.

    Args:
        X11: 3D points from view 1 [B, H, W, 3]
        X21: 3D points from view 2 [B, H, W, 3]
        idx_1_to_2_init: Initial indices [B, H*W]

    Returns:
        rays_with_grad: Ray image with gradients [B, H, W, 9]
        pts3d_norm: Normalized 3D points [B, H*W, 3]
        p_init: Initial pixel positions [B, H*W, 2]
    """
    b, h, w, _ = X11.shape

    # Compute ray image
    rays_img = normalize_rays(X11)  # [B, H, W, 3]

    # Convert to [B, C, H, W] for gradient computation
    rays_img_bchw = mx.transpose(rays_img, (0, 3, 1, 2))
    gx_img, gy_img = img_gradient(rays_img_bchw)

    # Concatenate rays and gradients: [B, 9, H, W]
    rays_with_grad_bchw = mx.concatenate([rays_img_bchw, gx_img, gy_img], axis=1)

    # Back to [B, H, W, 9]
    rays_with_grad = mx.transpose(rays_with_grad_bchw, (0, 2, 3, 1))

    # Normalize X21 points
    X21_vec = X21.reshape(b, -1, 3)
    pts3d_norm = normalize_rays(X21_vec)

    # Initial pixel positions
    if idx_1_to_2_init is None:
        idx_1_to_2_init = mx.broadcast_to(mx.arange(h * w)[None, :], (b, h * w))
    p_init = lin_to_pixel(idx_1_to_2_init, w).astype(mx.float32)

    return rays_with_grad, pts3d_norm, p_init


def bilinear_sample(img: mx.array, coords: mx.array) -> mx.array:
    """Vectorized bilinear sampling.

    Args:
        img: Image tensor [B, H, W, C]
        coords: Sampling coordinates [B, N, 2] (x, y)

    Returns:
        Sampled values [B, N, C]
    """
    b, h, w, c = img.shape
    n = coords.shape[1]

    # Clamp coordinates to valid range for interpolation
    x = mx.clip(coords[..., 0], 0, w - 1.001)
    y = mx.clip(coords[..., 1], 0, h - 1.001)

    # Integer and fractional parts
    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, w - 1)
    y1 = mx.minimum(y0 + 1, h - 1)

    fx = (x - x0.astype(mx.float32))[..., None]  # [B, N, 1]
    fy = (y - y0.astype(mx.float32))[..., None]  # [B, N, 1]

    # Flatten image for gather
    img_flat = img.reshape(b, h * w, c)  # [B, H*W, C]

    # Convert 2D indices to flat indices
    idx00 = y0 * w + x0  # [B, N]
    idx01 = y1 * w + x0
    idx10 = y0 * w + x1
    idx11 = y1 * w + x1

    # Gather values - process each batch element
    # Use take_along_axis for gathering
    result = mx.zeros((b, n, c), dtype=img.dtype)

    for bi in range(b):
        v00 = img_flat[bi, idx00[bi], :]  # [N, C]
        v01 = img_flat[bi, idx01[bi], :]
        v10 = img_flat[bi, idx10[bi], :]
        v11 = img_flat[bi, idx11[bi], :]

        # Bilinear interpolation
        interp = (
            (1 - fx[bi]) * (1 - fy[bi]) * v00
            + (1 - fx[bi]) * fy[bi] * v01
            + fx[bi] * (1 - fy[bi]) * v10
            + fx[bi] * fy[bi] * v11
        )
        result = result.at[bi].add(interp)

    return result


def iter_proj_mlx(
    rays_with_grad: mx.array,
    pts3d_norm: mx.array,
    p_init: mx.array,
    max_iter: int,
    lambda_init: float,
    convergence_thresh: float,
) -> tuple[mx.array, mx.array]:
    """Iterative projection matching (pure MLX implementation).

    Projects 3D points onto ray image using Levenberg-Marquardt optimization.

    Args:
        rays_with_grad: Ray image with gradients [B, H, W, 9]
        pts3d_norm: Normalized 3D points to project [B, N, 3]
        p_init: Initial pixel positions [B, N, 2]
        max_iter: Maximum iterations
        lambda_init: Initial LM damping factor
        convergence_thresh: Convergence threshold

    Returns:
        p: Final pixel positions [B, N, 2]
        valid: Validity mask [B, N]
    """
    b, h, w, _ = rays_with_grad.shape
    n = pts3d_norm.shape[1]

    p = p_init.astype(mx.float32)
    lam = lambda_init

    # Precompute damping matrix
    damping = lam * mx.eye(2, dtype=mx.float32)

    for _ in range(max_iter):
        # Sample rays and gradients using bilinear interpolation
        rays_sampled = bilinear_sample(rays_with_grad, p)  # [B, N, 9]

        # Extract rays and gradients
        rays = rays_sampled[..., :3]
        grad_x = rays_sampled[..., 3:6]
        grad_y = rays_sampled[..., 6:9]

        # Residual: r = ray_sampled - target_ray
        r = rays - pts3d_norm  # [B, N, 3]

        # Jacobian: J = [dr/dx, dr/dy]
        # J is [B, N, 3, 2]
        J = mx.stack([grad_x, grad_y], axis=-1)

        # Gauss-Newton with LM damping
        # (J^T J + lambda * I) * delta = -J^T r
        # J is [B, N, 3, 2] where 3 is residual dim and 2 is parameter dim

        # J^T J: [B, N, 2, 2] - contract over residual dimension (axis 2)
        JtJ = mx.einsum("bnki,bnkj->bnij", J, J)

        # Add damping [B, N, 2, 2]
        JtJ_damped = JtJ + damping[None, None, :, :]

        # J^T r: [B, N, 2] - contract over residual dimension
        Jtr = mx.einsum("bnki,bnk->bni", J, r)

        # Solve 2x2 system analytically
        # delta = -inv(JtJ_damped) @ Jtr
        a = JtJ_damped[..., 0, 0]
        b_val = JtJ_damped[..., 0, 1]
        c = JtJ_damped[..., 1, 0]
        d = JtJ_damped[..., 1, 1]

        det = a * d - b_val * c
        det = mx.maximum(det, 1e-10)
        inv_det = 1.0 / det

        # Compute inverse directly
        # inv = [[d, -b], [-c, a]] / det
        delta_x = -(d * Jtr[..., 0] - b_val * Jtr[..., 1]) * inv_det
        delta_y = -(-c * Jtr[..., 0] + a * Jtr[..., 1]) * inv_det
        delta = mx.stack([delta_x, delta_y], axis=-1)

        # Update
        p = p + delta

        # Check convergence (early stopping)
        delta_norm = mx.sqrt(mx.sum(delta * delta, axis=-1))
        max_delta = mx.max(delta_norm)
        mx.eval(max_delta)  # Force evaluation for convergence check
        if float(max_delta.item()) < convergence_thresh:
            break

    # Final clamp and validity check
    p_final = mx.stack(
        [
            mx.clip(p[..., 0], 0, w - 1),
            mx.clip(p[..., 1], 0, h - 1),
        ],
        axis=-1,
    )

    # Valid if within image bounds
    valid = (p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h)

    return p_final, valid


def match_iterative_proj(
    X11: mx.array,
    X21: mx.array,
    D11: mx.array,
    D21: mx.array,
    idx_1_to_2_init: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Match points using iterative projection.

    Args:
        X11: 3D points from view 1 [B, H, W, 3]
        X21: 3D points from view 2 [B, H, W, 3]
        D11: Descriptors from view 1 [B, H, W, D]
        D21: Descriptors from view 2 [B, H, W, D]
        idx_1_to_2_init: Initial indices [B, H*W]

    Returns:
        idx_1_to_2: Final indices [B, H*W]
        valid_match2: Validity mask [B, H*W, 1]
    """
    config = get_config()
    cfg = config["matching"]
    b, h, w = X21.shape[:3]

    # Prepare data
    rays_with_grad, pts3d_norm, p_init = prep_for_iter_proj(X11, X21, idx_1_to_2_init)

    # Try Metal GPU backend first (87x faster for large N)
    use_metal = cfg.get("use_metal", True)
    if use_metal:
        try:
            from mlx_mast3r_slam.backends.mpsgraph.kernels import iter_proj
            import numpy as np

            # Convert MLX to numpy for Metal kernel
            mx.eval(rays_with_grad, pts3d_norm, p_init)
            rays_np = np.array(rays_with_grad)
            pts_np = np.array(pts3d_norm)
            p_init_np = np.array(p_init)

            # Run Metal kernel
            p1_np, valid_np = iter_proj(
                rays_np, pts_np, p_init_np,
                cfg["max_iter"],
                cfg["lambda_init"],
                cfg["convergence_thresh"],
                use_metal=True,
            )

            # Convert back to MLX
            p1 = mx.array(p1_np)
            valid_proj = mx.array(valid_np)
        except Exception:
            # Fallback to MLX
            p1, valid_proj = iter_proj_mlx(
                rays_with_grad, pts3d_norm, p_init,
                cfg["max_iter"], cfg["lambda_init"], cfg["convergence_thresh"],
            )
    else:
        # Run pure MLX iterative projection
        p1, valid_proj = iter_proj_mlx(
            rays_with_grad, pts3d_norm, p_init,
            cfg["max_iter"], cfg["lambda_init"], cfg["convergence_thresh"],
        )

    # Refine matches using descriptor correlation (Metal accelerated)
    refine_radius = cfg.get("refine_radius", 3)
    refine_dilation = cfg.get("refine_dilation", 2)
    use_refine = cfg.get("use_refine", True)

    if use_refine and refine_radius > 0:
        p1_int = p1.astype(mx.int32)
        if use_metal:
            try:
                from mlx_mast3r_slam.backends.mpsgraph.kernels import refine_matches
                import numpy as np

                # Prepare descriptors
                D11_flat = D11.reshape(b, h, w, -1)
                D21_flat = D21.reshape(b, h * w, -1)

                mx.eval(D11_flat, D21_flat, p1_int)
                D11_np = np.array(D11_flat)
                D21_np = np.array(D21_flat)
                p1_np = np.array(p1_int)

                # Run Metal kernel
                p1_refined_np = refine_matches(
                    D11_np, D21_np, p1_np,
                    refine_radius, refine_dilation,
                    use_metal=True,
                )
                p1_int = mx.array(p1_refined_np)
            except Exception:
                # Fallback to unrefined positions
                pass
    else:
        p1_int = p1.astype(mx.int32)

    # Sample X11 at projected locations using vectorized gather
    X11_flat = X11.reshape(b, h * w, 3)  # [B, H*W, 3]

    # Compute linear indices for sampling
    y_idx = mx.clip(p1_int[..., 1], 0, h - 1)
    x_idx = mx.clip(p1_int[..., 0], 0, w - 1)
    lin_idx = y_idx * w + x_idx  # [B, N]

    # Gather X11 values at projected locations
    X11_sampled = mx.zeros((b, h * w, 3), dtype=X11.dtype)
    for bi in range(b):
        X11_sampled = X11_sampled.at[bi].add(X11_flat[bi, lin_idx[bi], :])

    # Compute distances for occlusion check
    X21_flat = X21.reshape(b, h * w, 3)
    dists = mx.sqrt(mx.sum((X11_sampled - X21_flat) ** 2, axis=-1))
    valid_dists = dists < cfg["dist_thresh"]

    valid_proj = valid_proj & valid_dists

    # Convert to linear indices
    idx_1_to_2 = pixel_to_lin(p1_int, w)

    return idx_1_to_2, valid_proj[:, :, None]


def refine_matches_correlation(
    D11: mx.array,
    D21: mx.array,
    p1: mx.array,
    radius: int,
) -> mx.array:
    """Refine matches using descriptor correlation in a local window.

    Args:
        D11: Descriptors from view 1 [B, H, W, D]
        D21: Descriptors from view 2 [B, H*W, D]
        p1: Current pixel positions [B, N, 2]
        radius: Search radius

    Returns:
        Refined pixel positions [B, N, 2]
    """
    b, h, w, d = D11.shape
    n = p1.shape[1]

    p_refined = p1.astype(mx.int32)

    for bi in range(b):
        for ni in range(n):
            cx, cy = int(p1[bi, ni, 0].item()), int(p1[bi, ni, 1].item())

            # Get query descriptor
            query_desc = D21[bi, ni, :]  # [D]

            best_score = -float("inf")
            best_x, best_y = cx, cy

            # Search in window
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = cy + dy
                    nx = cx + dx

                    if 0 <= nx < w and 0 <= ny < h:
                        ref_desc = D11[bi, ny, nx, :]
                        # Cosine similarity
                        score = mx.sum(query_desc * ref_desc)
                        if score > best_score:
                            best_score = score
                            best_x, best_y = nx, ny

            p_refined = p_refined.at[bi, ni, 0].add(best_x - cx)
            p_refined = p_refined.at[bi, ni, 1].add(best_y - cy)

    return p_refined
