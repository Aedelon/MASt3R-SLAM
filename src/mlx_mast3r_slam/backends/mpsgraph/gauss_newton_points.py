# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Gauss-Newton optimization for 3D point alignment (numpy fallback)."""

from __future__ import annotations

import numpy as np

from .sim3_ops import (
    sim3_relative,
    sim3_act,
    retract_sim3,
    huber_weight,
    quat_rotate,
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
) -> np.ndarray:
    """Gauss-Newton optimization for 3D point alignment.

    Point-based mode: uses direct 3D point error without ray normalization.
    """
    num_kf = Twc.shape[0]
    num_edges = len(ii)
    num_pts = Xs.shape[1]

    if num_edges == 0 or num_kf <= pin:
        return Twc.copy()

    unique_kf = np.unique(np.concatenate([ii, jj]))
    num_unique = len(unique_kf)

    if num_unique <= pin:
        return Twc.copy()

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

    sigma_inv = 1.0 / sigma_point

    # Optimization loop
    for iteration in range(max_iter):
        dim = 7 * num_free
        H = np.zeros((dim, dim), dtype=np.float64)
        g = np.zeros(dim, dtype=np.float64)

        for edge_idx in range(num_edges):
            ix = int(ii[edge_idx])
            jx = int(jj[edge_idx])

            i_local = kf_to_local.get(ix, -pin - 1)
            j_local = kf_to_local.get(jx, -pin - 1)

            if i_local < 0 and j_local < 0:
                continue

            # Relative transformation
            tij, qij, sij = sim3_relative(t[ix], q[ix], s[ix], t[jx], q[jx], s[jx])

            # Get correspondences
            idx_corr = idx_ii2jj[edge_idx]
            valid = valid_match[edge_idx]
            q_conf = Q[edge_idx]
            ci = Cs[ix, idx_corr]
            cj = Cs[jx]

            mask = valid & (q_conf > Q_thresh) & (ci > C_thresh) & (cj > C_thresh)
            valid_idx = np.where(mask)[0]

            if len(valid_idx) == 0:
                continue

            # Get valid points
            Xi = Xs[ix, idx_corr[valid_idx]]
            Xj = Xs[jx, valid_idx]
            conf = q_conf[valid_idx]

            # Transform Xj to frame i
            Xj_Ci = quat_rotate(qij[None], Xj) * sij + tij

            # Residual: direct 3D error
            err = Xj_Ci - Xi

            # Scale-invariant weighting
            dist = np.linalg.norm(Xi, axis=-1) + 1e-6
            scale_factor = 1.0 / dist

            # Weights
            sqrt_w = sigma_inv * np.sqrt(conf) * scale_factor
            weighted_err = sqrt_w[:, None] * err
            hub_w = huber_weight(weighted_err)
            w = hub_w * (sqrt_w[:, None] ** 2)

            n_valid = len(valid_idx)

            # Jacobian of X w.r.t. pose
            Ji = np.zeros((n_valid, 3, 7), dtype=np.float64)
            Ji[:, 0, 0] = 1.0
            Ji[:, 1, 1] = 1.0
            Ji[:, 2, 2] = 1.0
            Ji[:, 0, 4] = Xj_Ci[:, 2]
            Ji[:, 0, 5] = -Xj_Ci[:, 1]
            Ji[:, 1, 3] = -Xj_Ci[:, 2]
            Ji[:, 1, 5] = Xj_Ci[:, 0]
            Ji[:, 2, 3] = Xj_Ci[:, 1]
            Ji[:, 2, 4] = -Xj_Ci[:, 0]
            Ji[:, 0, 6] = Xj_Ci[:, 0]
            Ji[:, 1, 6] = Xj_Ci[:, 1]
            Ji[:, 2, 6] = Xj_Ci[:, 2]

            # Compute Jj via adjoint
            s_inv = 1.0 / s[ix]
            qi_inv = np.array([-q[ix, 0], -q[ix, 1], -q[ix, 2], q[ix, 3]])
            Jj = np.zeros_like(Ji)

            for k in range(n_valid):
                for coord in range(3):
                    Ji_vec = Ji[k, coord, :3]
                    Ji_rot = Ji[k, coord, 3:6]
                    Jj[k, coord, :3] = s_inv * quat_rotate(qi_inv, Ji_vec)
                    Jj[k, coord, 3:6] = quat_rotate(qi_inv, Ji_rot)
                    Jj[k, coord, 6] = Ji[k, coord, 6]

            Ji = -Jj

            # Weighted Jacobians
            wJi = w[:, :, None] * Ji
            wJj = w[:, :, None] * Jj

            # Accumulate Hessian blocks
            if i_local >= 0:
                for coord in range(3):
                    JiT = Ji[:, coord, :]
                    wJiT = wJi[:, coord, :]
                    Hii = JiT.T @ wJiT
                    H[i_local * 7 : (i_local + 1) * 7, i_local * 7 : (i_local + 1) * 7] += Hii
                    g[i_local * 7 : (i_local + 1) * 7] += wJiT.T @ err[:, coord]

            if j_local >= 0:
                for coord in range(3):
                    JjT = Jj[:, coord, :]
                    wJjT = wJj[:, coord, :]
                    Hjj = JjT.T @ wJjT
                    H[j_local * 7 : (j_local + 1) * 7, j_local * 7 : (j_local + 1) * 7] += Hjj
                    g[j_local * 7 : (j_local + 1) * 7] += wJjT.T @ err[:, coord]

            if i_local >= 0 and j_local >= 0:
                for coord in range(3):
                    JiT = Ji[:, coord, :]
                    wJjT = wJj[:, coord, :]
                    Hij = JiT.T @ wJjT
                    H[i_local * 7 : (i_local + 1) * 7, j_local * 7 : (j_local + 1) * 7] += Hij
                    H[j_local * 7 : (j_local + 1) * 7, i_local * 7 : (i_local + 1) * 7] += Hij.T

        # Regularization and solve
        H += np.eye(dim, dtype=np.float64) * 1e-6

        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break

        if np.linalg.norm(dx) < delta_thresh:
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
