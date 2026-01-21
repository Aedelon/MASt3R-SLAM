# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Gauss-Newton optimization for ray-based pose alignment (vectorized).

Implements the core optimization loop for MASt3R-SLAM with vectorized operations
for performance on Apple Silicon.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .sim3_ops import (
    sim3_relative,
    sim3_act,
    retract_sim3,
    huber_weight,
    quat_rotate,
)


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
) -> np.ndarray:
    """Gauss-Newton optimization for ray alignment (vectorized).

    Args:
        Twc: Poses [num_kf, 8] as (tx, ty, tz, qx, qy, qz, qw, scale)
        Xs: 3D points [num_kf, num_pts, 3]
        Cs: Confidences [num_kf, num_pts, 1] or [num_kf, num_pts]
        ii: Source keyframe indices [num_edges]
        jj: Target keyframe indices [num_edges]
        idx_ii2jj: Point correspondences [num_edges, num_pts]
        valid_match: Match validity [num_edges, num_pts, 1] or [num_edges, num_pts]
        Q: Match confidence [num_edges, num_pts, 1] or [num_edges, num_pts]
        sigma_ray: Ray residual standard deviation
        sigma_dist: Distance residual standard deviation (unused in ray mode)
        C_thresh: Confidence threshold
        Q_thresh: Match quality threshold
        max_iter: Maximum iterations
        delta_thresh: Convergence threshold
        pin: Number of poses to fix (usually 1)

    Returns:
        Updated Twc poses [num_kf, 8]
    """
    num_kf = Twc.shape[0]
    num_edges = len(ii)
    num_pts = Xs.shape[1]

    if num_edges == 0 or num_kf <= pin:
        return Twc.copy()

    # Get unique keyframe indices
    unique_kf = np.unique(np.concatenate([ii, jj]))
    num_unique = len(unique_kf)

    if num_unique <= pin:
        return Twc.copy()

    # Create index mapping
    kf_to_local = {int(kf): i - pin for i, kf in enumerate(unique_kf)}
    num_free = num_unique - pin

    if num_free <= 0:
        return Twc.copy()

    # Normalize input shapes
    if valid_match.ndim == 3:
        valid_match = valid_match[..., 0]
    if Q.ndim == 3:
        Q = Q[..., 0]
    if Cs.ndim == 3:
        Cs = Cs[..., 0]

    # Extract pose components
    t = Twc[:, :3].astype(np.float64)
    q = Twc[:, 3:7].astype(np.float64)
    s = Twc[:, 7].astype(np.float64)

    sigma_inv = 1.0 / sigma_ray

    # Optimization loop
    for iteration in range(max_iter):
        # Initialize dense Hessian blocks and gradient
        dim = 7 * num_free
        H = np.zeros((dim, dim), dtype=np.float64)
        g = np.zeros(dim, dtype=np.float64)

        # Process each edge (vectorized over points)
        for edge_idx in range(num_edges):
            ix = int(ii[edge_idx])
            jx = int(jj[edge_idx])

            i_local = kf_to_local.get(ix, -pin - 1)
            j_local = kf_to_local.get(jx, -pin - 1)

            if i_local < 0 and j_local < 0:
                continue

            # Compute relative transformation
            tij, qij, sij = sim3_relative(t[ix], q[ix], s[ix], t[jx], q[jx], s[jx])

            # Get correspondences
            idx_corr = idx_ii2jj[edge_idx]
            valid = valid_match[edge_idx]

            # Get confidence weights
            q_conf = Q[edge_idx]
            ci = Cs[ix, idx_corr]
            cj = Cs[jx]

            # Build validity mask
            mask = valid & (q_conf > Q_thresh) & (ci > C_thresh) & (cj > C_thresh)
            valid_idx = np.where(mask)[0]

            if len(valid_idx) == 0:
                continue

            # Get valid points
            Xi = Xs[ix, idx_corr[valid_idx]]  # [N_valid, 3]
            Xj = Xs[jx, valid_idx]  # [N_valid, 3]
            conf = q_conf[valid_idx]  # [N_valid]

            # Transform Xj to frame i: Xj_Ci = sij * R(qij) @ Xj + tij
            Xj_Ci = quat_rotate(qij[None], Xj) * sij + tij  # [N_valid, 3]

            # Residual
            err = Xj_Ci - Xi  # [N_valid, 3]

            # Weights
            sqrt_w = sigma_inv * np.sqrt(conf)  # [N_valid]
            weighted_err = sqrt_w[:, None] * err  # [N_valid, 3]

            # Huber weights per component
            hub_w = huber_weight(weighted_err)  # [N_valid, 3]
            w = hub_w * (sqrt_w[:, None] ** 2)  # [N_valid, 3]

            # Jacobians (vectorized)
            # J_i = [I_3 | [Xj_Ci]_x | Xj_Ci] for each coordinate
            # J_j = -Ad^{-1}_Ti @ J_i

            n_valid = len(valid_idx)

            # For each residual coordinate, compute contribution to H and g
            # We'll compute the full 14x14 Jacobian outer product per point

            # Build Jacobian for pose i: [N_valid, 3, 7]
            Ji = np.zeros((n_valid, 3, 7), dtype=np.float64)

            # Translation part (identity)
            Ji[:, 0, 0] = 1.0
            Ji[:, 1, 1] = 1.0
            Ji[:, 2, 2] = 1.0

            # Rotation part: [Xj_Ci]_x
            Ji[:, 0, 3] = 0
            Ji[:, 0, 4] = Xj_Ci[:, 2]
            Ji[:, 0, 5] = -Xj_Ci[:, 1]

            Ji[:, 1, 3] = -Xj_Ci[:, 2]
            Ji[:, 1, 4] = 0
            Ji[:, 1, 5] = Xj_Ci[:, 0]

            Ji[:, 2, 3] = Xj_Ci[:, 1]
            Ji[:, 2, 4] = -Xj_Ci[:, 0]
            Ji[:, 2, 5] = 0

            # Scale part
            Ji[:, 0, 6] = Xj_Ci[:, 0]
            Ji[:, 1, 6] = Xj_Ci[:, 1]
            Ji[:, 2, 6] = Xj_Ci[:, 2]

            # Compute Jj = -Ad^{-1}_Ti @ Ji (simplified version)
            # Ad^{-1}_T has form that transforms the Jacobian
            # For simplicity, we use: Jj ≈ -s_i^{-1} * R_i^T @ Ji (approximate)
            s_inv = 1.0 / s[ix]
            Jj = np.zeros_like(Ji)

            # Rotation part of adjoint
            for k in range(n_valid):
                for coord in range(3):
                    # Rotate Ji by R_i^{-1}
                    Ji_vec = Ji[k, coord, :3]  # Translation Jacobian
                    Ji_rot = Ji[k, coord, 3:6]  # Rotation Jacobian

                    # Simplified adjoint action
                    Jj[k, coord, :3] = s_inv * quat_rotate(
                        np.array([-q[ix, 0], -q[ix, 1], -q[ix, 2], q[ix, 3]]),
                        Ji_vec,
                    )
                    Jj[k, coord, 3:6] = quat_rotate(
                        np.array([-q[ix, 0], -q[ix, 1], -q[ix, 2], q[ix, 3]]),
                        Ji_rot,
                    )
                    Jj[k, coord, 6] = Ji[k, coord, 6]

            Ji = -Jj  # Sign convention

            # Weighted Jacobians
            # w: [N_valid, 3], Ji: [N_valid, 3, 7]
            wJi = w[:, :, None] * Ji  # [N_valid, 3, 7]
            wJj = w[:, :, None] * Jj  # [N_valid, 3, 7]

            # Accumulate Hessian blocks
            # Hii = sum_k sum_coord wJi[k,coord] @ Ji[k,coord].T
            if i_local >= 0:
                # Hii block
                for coord in range(3):
                    JiT = Ji[:, coord, :]  # [N_valid, 7]
                    wJiT = wJi[:, coord, :]  # [N_valid, 7]
                    Hii = JiT.T @ wJiT  # [7, 7]
                    H[i_local * 7 : (i_local + 1) * 7, i_local * 7 : (i_local + 1) * 7] += Hii

                    # Gradient
                    g[i_local * 7 : (i_local + 1) * 7] += wJiT.T @ err[:, coord]

            if j_local >= 0:
                # Hjj block
                for coord in range(3):
                    JjT = Jj[:, coord, :]  # [N_valid, 7]
                    wJjT = wJj[:, coord, :]  # [N_valid, 7]
                    Hjj = JjT.T @ wJjT  # [7, 7]
                    H[j_local * 7 : (j_local + 1) * 7, j_local * 7 : (j_local + 1) * 7] += Hjj

                    # Gradient
                    g[j_local * 7 : (j_local + 1) * 7] += wJjT.T @ err[:, coord]

            if i_local >= 0 and j_local >= 0:
                # Hij block (off-diagonal)
                for coord in range(3):
                    JiT = Ji[:, coord, :]
                    wJjT = wJj[:, coord, :]
                    Hij = JiT.T @ wJjT  # [7, 7]
                    H[i_local * 7 : (i_local + 1) * 7, j_local * 7 : (j_local + 1) * 7] += Hij
                    H[j_local * 7 : (j_local + 1) * 7, i_local * 7 : (i_local + 1) * 7] += Hij.T

        # Add regularization
        H += np.eye(dim, dtype=np.float64) * 1e-6

        # Solve H @ dx = -g
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break

        # Check convergence
        delta_norm = np.linalg.norm(dx)
        if delta_norm < delta_thresh:
            break

        # Update poses
        for i, kf_idx in enumerate(unique_kf[pin:]):
            kf_idx = int(kf_idx)
            xi = dx[i * 7 : (i + 1) * 7]

            t_new, q_new, s_new = retract_sim3(xi, t[kf_idx], q[kf_idx], s[kf_idx])

            t[kf_idx] = t_new
            q[kf_idx] = q_new
            s[kf_idx] = s_new

    # Reconstruct Twc
    Twc_new = np.concatenate([t, q, s[:, None]], axis=-1).astype(np.float32)
    return Twc_new
