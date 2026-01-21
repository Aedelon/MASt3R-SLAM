# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Geometry utilities for MLX-MASt3R-SLAM."""

from __future__ import annotations

import mlx.core as mx

from mlx_mast3r_slam.liegroups import Sim3


def skew_sym(x: mx.array) -> mx.array:
    """Create skew-symmetric matrix from vector.

    Args:
        x: Vector [..., 3]

    Returns:
        Skew-symmetric matrix [..., 3, 3]
    """
    batch_shape = x.shape[:-1]
    x_flat = x.reshape(-1, 3)

    result = mx.zeros((x_flat.shape[0], 3, 3), dtype=x.dtype)
    result = result.at[:, 0, 1].add(-x_flat[:, 2])
    result = result.at[:, 0, 2].add(x_flat[:, 1])
    result = result.at[:, 1, 0].add(x_flat[:, 2])
    result = result.at[:, 1, 2].add(-x_flat[:, 0])
    result = result.at[:, 2, 0].add(-x_flat[:, 1])
    result = result.at[:, 2, 1].add(x_flat[:, 0])

    return result.reshape(batch_shape + (3, 3))


def point_to_dist(X: mx.array) -> mx.array:
    """Compute distance (norm) of 3D points.

    Args:
        X: 3D points [..., 3]

    Returns:
        Distances [..., 1]
    """
    return mx.sqrt(mx.sum(X * X, axis=-1, keepdims=True) + 1e-10)


def point_to_ray_dist(
    X: mx.array, jacobian: bool = False
) -> mx.array | tuple[mx.array, mx.array]:
    """Convert 3D points to ray-distance representation.

    Args:
        X: 3D points [..., 3]
        jacobian: Whether to compute jacobian

    Returns:
        rd: Ray-distance [..., 4] with [rx, ry, rz, d]
        drd_dX: Jacobian [..., 4, 3] (if jacobian=True)
    """
    d = point_to_dist(X)
    d_inv = 1.0 / d
    r = d_inv * X
    rd = mx.concatenate([r, d], axis=-1)

    if not jacobian:
        return rd

    # Jacobian computation
    batch_shape = X.shape[:-1]
    n = mx.prod(mx.array(batch_shape)).item() if batch_shape else 1

    X_flat = X.reshape(-1, 3)
    d_inv_flat = d_inv.reshape(-1, 1)
    d_inv_2 = d_inv_flat**2
    r_flat = r.reshape(-1, 3)

    # dr/dX = (1/d) * (I - (1/d^2) * X @ X.T)
    I = mx.eye(3, dtype=X.dtype)
    I_batch = mx.broadcast_to(I, (n, 3, 3))

    # X @ X.T term
    XXT = X_flat[:, :, None] @ X_flat[:, None, :]

    dr_dX = d_inv_flat[:, :, None] * (I_batch - d_inv_2[:, :, None] * XXT)

    # dd/dX = r.T (normalized direction)
    dd_dX = r_flat[:, None, :]

    # Stack: [dr_dX (3x3), dd_dX (1x3)] -> (4x3)
    drd_dX = mx.concatenate([dr_dX, dd_dX], axis=1)
    drd_dX = drd_dX.reshape(batch_shape + (4, 3))

    return rd, drd_dX


def act_Sim3(
    X: Sim3, pC: mx.array, jacobian: bool = False
) -> mx.array | tuple[mx.array, mx.array]:
    """Transform points by Sim3 and optionally compute jacobian.

    Args:
        X: Sim3 transformation
        pC: Points in camera frame [..., 3]
        jacobian: Whether to compute jacobian w.r.t. Sim3 tangent

    Returns:
        pW: Transformed points [..., 3]
        dpW_dX: Jacobian [..., 3, 7] (if jacobian=True)
    """
    pW = X.act(pC)

    if not jacobian:
        return pW

    # Jacobian of Sim3.act w.r.t. tangent vector [v, omega, sigma]
    # p_W = s * R * p_C + t
    # dp/dv = I (3x3)
    # dp/domega = -[p_W]_x (skew-symmetric)
    # dp/dsigma = p_W

    batch_shape = pW.shape[:-1]

    # dp/dt = I
    dpC_dt = mx.eye(3, dtype=pW.dtype)
    dpC_dt = mx.broadcast_to(dpC_dt, batch_shape + (3, 3))

    # dp/dR = -skew(pW)
    dpC_dR = -skew_sym(pW)

    # dp/dsigma = pW (the transformed point)
    # Since sigma = log(s), ds/dsigma = s, and dp/ds = R*p_C
    # dp/dsigma = s * R * p_C = pW - t, but original code uses pW directly
    dpc_ds = pW[..., None]  # [..., 3, 1]

    # Combine: [dt (3), dR (3), ds (1)] -> [..., 3, 7]
    J = mx.concatenate([dpC_dt, dpC_dR, dpc_ds], axis=-1)

    return pW, J


def decompose_K(K: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Decompose intrinsic matrix K into components.

    Args:
        K: Intrinsic matrix [3, 3]

    Returns:
        fx, fy, cx, cy
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy


def project_calib(
    P: mx.array,
    K: mx.array,
    img_size: tuple[int, int],
    jacobian: bool = False,
    border: int = 0,
    z_eps: float = 0.0,
) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array]:
    """Project 3D points using calibrated camera.

    Args:
        P: 3D points [..., 3]
        K: Intrinsic matrix [3, 3]
        img_size: Image size (height, width)
        jacobian: Whether to compute jacobian
        border: Border margin for valid check
        z_eps: Minimum z for valid projection

    Returns:
        pz: Projected points with log-depth [..., 3] as [u, v, log(z)]
        valid: Validity mask [..., 1]
        dpz_dP: Jacobian [..., 3, 3] (if jacobian=True)
    """
    batch_shape = P.shape[:-1]
    h, w = img_size

    # Project: p = K @ P, then normalize
    # Broadcast K to batch shape
    P_h = P[..., None]  # [..., 3, 1]
    p = (K @ P_h)[..., 0]  # [..., 3]
    z = p[..., 2:3]
    p_norm = p / (z + 1e-10)
    uv = p_norm[..., :2]

    u = uv[..., 0:1]
    v = uv[..., 1:2]

    # Validity checks
    valid_u = (u > border) & (u < w - 1 - border)
    valid_v = (v > border) & (v < h - 1 - border)
    valid_z = P[..., 2:3] > z_eps
    valid = valid_u & valid_v & valid_z

    # Log-depth
    logz = mx.log(P[..., 2:3] + 1e-10)
    logz = mx.where(valid_z, logz, mx.zeros_like(logz))

    pz = mx.concatenate([uv, logz], axis=-1)

    if not jacobian:
        return pz, valid

    # Jacobian computation
    fx, fy, cx, cy = decompose_K(K)
    x, y, z_p = P[..., 0], P[..., 1], P[..., 2]
    z_inv = 1.0 / (z_p + 1e-10)

    dpz_dP = mx.zeros(batch_shape + (3, 3), dtype=P.dtype)

    # du/dx = fx/z, du/dy = 0, du/dz = -fx*x/z^2
    dpz_dP = dpz_dP.at[..., 0, 0].add(fx * z_inv)
    dpz_dP = dpz_dP.at[..., 0, 2].add(-fx * x * z_inv * z_inv)

    # dv/dx = 0, dv/dy = fy/z, dv/dz = -fy*y/z^2
    dpz_dP = dpz_dP.at[..., 1, 1].add(fy * z_inv)
    dpz_dP = dpz_dP.at[..., 1, 2].add(-fy * y * z_inv * z_inv)

    # d(logz)/dx = 0, d(logz)/dy = 0, d(logz)/dz = 1/z
    dpz_dP = dpz_dP.at[..., 2, 2].add(z_inv)

    return pz, dpz_dP, valid


def backproject(p: mx.array, z: mx.array, K: mx.array) -> mx.array:
    """Backproject 2D points to 3D using depth.

    Args:
        p: 2D pixel coordinates [..., 2]
        z: Depth values [..., 1]
        K: Intrinsic matrix [3, 3]

    Returns:
        P: 3D points [..., 3]
    """
    fx, fy, cx, cy = decompose_K(K)

    x = (p[..., 0:1] - cx) / fx * z
    y = (p[..., 1:2] - cy) / fy * z
    P = mx.concatenate([x, y, z], axis=-1)

    return P


def get_pixel_coords(
    batch_size: int,
    img_size: tuple[int, int],
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Generate pixel coordinate grid.

    Args:
        batch_size: Batch size
        img_size: Image size (height, width)
        dtype: Data type

    Returns:
        uv: Pixel coordinates [batch, height, width, 2]
    """
    h, w = img_size
    u = mx.arange(w, dtype=dtype)
    v = mx.arange(h, dtype=dtype)
    v_grid, u_grid = mx.meshgrid(v, u, indexing="ij")
    uv = mx.stack([u_grid, v_grid], axis=-1)
    uv = mx.broadcast_to(uv[None], (batch_size, h, w, 2))
    return uv


def constrain_points_to_ray(
    img_size: tuple[int, int],
    Xs: mx.array,
    K: mx.array,
) -> mx.array:
    """Constrain 3D points to lie on camera rays (for calibrated mode).

    Args:
        img_size: Image size (height, width)
        Xs: 3D points [batch, H*W, 3]
        K: Intrinsic matrix [3, 3]

    Returns:
        Constrained 3D points [batch, H*W, 3]
    """
    batch_size = Xs.shape[0]
    n_points = Xs.shape[1]
    h, w = img_size

    # Get pixel coordinates
    uv = get_pixel_coords(batch_size, img_size, dtype=Xs.dtype)
    uv = uv.reshape(batch_size, -1, 2)

    # Extract depth from current points
    z = Xs[..., 2:3]

    # Backproject with fixed depth to constrain to ray
    Xs_constrained = backproject(uv, z, K)

    return Xs_constrained


def normalize_rays(X: mx.array) -> mx.array:
    """Normalize 3D points to unit rays.

    Args:
        X: 3D points [..., 3]

    Returns:
        Normalized rays [..., 3]
    """
    norm = mx.sqrt(mx.sum(X * X, axis=-1, keepdims=True) + 1e-10)
    return X / norm


def cartesian_to_spherical(P: mx.array) -> mx.array:
    """Convert cartesian to spherical coordinates.

    Args:
        P: Cartesian points [..., 3]

    Returns:
        Spherical coordinates [..., 3] as [r, phi, theta]
    """
    r = mx.sqrt(mx.sum(P * P, axis=-1, keepdims=True) + 1e-10)
    x, y, z = P[..., 0:1], P[..., 1:2], P[..., 2:3]
    phi = mx.arctan2(y, x)
    theta = mx.arccos(mx.clip(z / r, -1.0, 1.0))
    return mx.concatenate([r, phi, theta], axis=-1)


def spherical_to_cartesian(spherical: mx.array) -> mx.array:
    """Convert spherical to cartesian coordinates.

    Args:
        spherical: Spherical coordinates [..., 3] as [r, phi, theta]

    Returns:
        Cartesian points [..., 3]
    """
    r = spherical[..., 0:1]
    phi = spherical[..., 1:2]
    theta = spherical[..., 2:3]

    x = r * mx.sin(theta) * mx.cos(phi)
    y = r * mx.sin(theta) * mx.sin(phi)
    z = r * mx.cos(theta)

    return mx.concatenate([x, y, z], axis=-1)
