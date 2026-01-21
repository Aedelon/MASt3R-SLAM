# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Gauss-Newton optimization for calibrated projection residuals (numpy fallback)."""

from __future__ import annotations

import numpy as np

from .sim3_ops import (
    sim3_relative,
    sim3_act,
    retract_sim3,
    huber_weight,
    quat_rotate,
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
) -> np.ndarray:
    """Gauss-Newton optimization for calibrated projection residuals.

    Residuals: (u_proj - u_obs, v_proj - v_obs, log(z_proj) - log(z_obs))
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

    # Extract intrinsics
    if K.shape == (3, 3):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        fx, fy, cx, cy = K.flatten()[:4]

    img_width, img_height = img_size
    sigma_pixel_inv = 1.0 / sigma_pixel
    sigma_depth_inv = 1.0 / sigma_depth

    # Extract pose components
    t = Twc[:, :3].astype(np.float64)
    q = Twc[:, 3:7].astype(np.float64)
    s = Twc[:, 7].astype(np.float64)

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

            # Check depth validity
            depth_valid = (Xj_Ci[:, 2] > z_eps) & (Xi[:, 2] > z_eps)
            if not np.any(depth_valid):
                continue

            valid_idx2 = np.where(depth_valid)[0]
            Xi = Xi[valid_idx2]
            Xj_Ci = Xj_Ci[valid_idx2]
            conf = conf[valid_idx2]

            n_valid = len(valid_idx2)

            # Project both points
            z_inv_j = 1.0 / Xj_Ci[:, 2]
            z_inv_i = 1.0 / Xi[:, 2]

            proj_j_u = fx * Xj_Ci[:, 0] * z_inv_j + cx
            proj_j_v = fy * Xj_Ci[:, 1] * z_inv_j + cy
            proj_i_u = fx * Xi[:, 0] * z_inv_i + cx
            proj_i_v = fy * Xi[:, 1] * z_inv_i + cy

            # Check bounds
            in_bounds = (
                (proj_j_u >= pixel_border)
                & (proj_j_u < img_width - pixel_border)
                & (proj_j_v >= pixel_border)
                & (proj_j_v < img_height - pixel_border)
            )
            if not np.any(in_bounds):
                continue

            valid_idx3 = np.where(in_bounds)[0]
            Xj_Ci = Xj_Ci[valid_idx3]
            Xi = Xi[valid_idx3]
            conf = conf[valid_idx3]
            proj_j_u = proj_j_u[valid_idx3]
            proj_j_v = proj_j_v[valid_idx3]
            proj_i_u = proj_i_u[valid_idx3]
            proj_i_v = proj_i_v[valid_idx3]
            z_inv_j = z_inv_j[valid_idx3]

            n_valid = len(valid_idx3)

            # Residuals
            err_u = (proj_j_u - proj_i_u) * sigma_pixel_inv
            err_v = (proj_j_v - proj_i_v) * sigma_pixel_inv
            err_z = (np.log(Xj_Ci[:, 2]) - np.log(Xi[:, 2])) * sigma_depth_inv

            err = np.stack([err_u, err_v, err_z], axis=-1)

            # Weights
            sqrt_w = np.sqrt(conf)
            weighted_err = sqrt_w[:, None] * err
            hub_w = huber_weight(weighted_err)
            w = hub_w * (sqrt_w[:, None] ** 2)

            # Jacobian of projection w.r.t. 3D point
            z_inv2_j = z_inv_j**2
            dproj_dX = np.zeros((n_valid, 3, 3), dtype=np.float64)
            dproj_dX[:, 0, 0] = fx * z_inv_j * sigma_pixel_inv
            dproj_dX[:, 0, 2] = -fx * Xj_Ci[:, 0] * z_inv2_j * sigma_pixel_inv
            dproj_dX[:, 1, 1] = fy * z_inv_j * sigma_pixel_inv
            dproj_dX[:, 1, 2] = -fy * Xj_Ci[:, 1] * z_inv2_j * sigma_pixel_inv
            dproj_dX[:, 2, 2] = z_inv_j * sigma_depth_inv

            # Jacobian of X w.r.t. pose
            Ji_X = np.zeros((n_valid, 3, 7), dtype=np.float64)
            Ji_X[:, 0, 0] = 1.0
            Ji_X[:, 1, 1] = 1.0
            Ji_X[:, 2, 2] = 1.0
            Ji_X[:, 0, 4] = Xj_Ci[:, 2]
            Ji_X[:, 0, 5] = -Xj_Ci[:, 1]
            Ji_X[:, 1, 3] = -Xj_Ci[:, 2]
            Ji_X[:, 1, 5] = Xj_Ci[:, 0]
            Ji_X[:, 2, 3] = Xj_Ci[:, 1]
            Ji_X[:, 2, 4] = -Xj_Ci[:, 0]
            Ji_X[:, 0, 6] = Xj_Ci[:, 0]
            Ji_X[:, 1, 6] = Xj_Ci[:, 1]
            Ji_X[:, 2, 6] = Xj_Ci[:, 2]

            # Compute Jj via adjoint
            s_inv = 1.0 / s[ix]
            qi_inv = np.array([-q[ix, 0], -q[ix, 1], -q[ix, 2], q[ix, 3]])
            Jj_X = np.zeros_like(Ji_X)

            for k in range(n_valid):
                for coord in range(3):
                    Ji_vec = Ji_X[k, coord, :3]
                    Ji_rot = Ji_X[k, coord, 3:6]
                    Jj_X[k, coord, :3] = s_inv * quat_rotate(qi_inv, Ji_vec)
                    Jj_X[k, coord, 3:6] = quat_rotate(qi_inv, Ji_rot)
                    Jj_X[k, coord, 6] = Ji_X[k, coord, 6]

            Ji_X = -Jj_X

            # Chain rule
            Ji = np.einsum("nrc,nck->nrk", dproj_dX, Ji_X)
            Jj = np.einsum("nrc,nck->nrk", dproj_dX, Jj_X)

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
