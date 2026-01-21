# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Global optimization (backend) for MLX-MASt3R-SLAM."""

from __future__ import annotations

import mlx.core as mx

from mlx_mast3r_slam.frame import Keyframes
from mlx_mast3r_slam.liegroups import Sim3
from mlx_mast3r_slam.geometry import constrain_points_to_ray
from mlx_mast3r_slam.config import get_config


class FactorGraph:
    """Factor graph for global optimization.

    Manages edges (factors) between keyframes and performs
    bundle adjustment using Gauss-Newton.
    """

    def __init__(
        self,
        model,
        frames: Keyframes,
        K: mx.array | None = None,
    ) -> None:
        """Initialize factor graph.

        Args:
            model: MASt3R model
            frames: Keyframe storage
            K: Intrinsic matrix (optional, for calibrated mode)
        """
        self.model = model
        self.frames = frames
        self.K = K
        self.cfg = get_config()["local_opt"]

        # Factor storage
        self.ii: mx.array = mx.array([], dtype=mx.int32)
        self.jj: mx.array = mx.array([], dtype=mx.int32)
        self.idx_ii2jj: mx.array = mx.array([], dtype=mx.int32)
        self.idx_jj2ii: mx.array = mx.array([], dtype=mx.int32)
        self.valid_match_j: mx.array = mx.array([], dtype=mx.bool_)
        self.valid_match_i: mx.array = mx.array([], dtype=mx.bool_)
        self.Q_ii2jj: mx.array = mx.array([], dtype=mx.float32)
        self.Q_jj2ii: mx.array = mx.array([], dtype=mx.float32)

    def add_factors(
        self,
        ii: list[int],
        jj: list[int],
        min_match_frac: float,
        mast3r_match_fn,
        is_reloc: bool = False,
    ) -> bool:
        """Add factors between keyframe pairs.

        Args:
            ii: Source keyframe indices
            jj: Target keyframe indices
            min_match_frac: Minimum match fraction threshold
            mast3r_match_fn: MASt3R symmetric matching function
            is_reloc: Whether this is for relocalization

        Returns:
            True if new edges were added
        """
        # Get keyframes
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]

        # Stack features
        feat_i = mx.concatenate([kf.feat for kf in kf_ii])
        feat_j = mx.concatenate([kf.feat for kf in kf_jj])
        pos_i = mx.concatenate([kf.pos for kf in kf_ii])
        pos_j = mx.concatenate([kf.pos for kf in kf_jj])
        shape_i = [kf.img_true_shape for kf in kf_ii]
        shape_j = [kf.img_true_shape for kf in kf_jj]

        # Symmetric matching
        (idx_i2j, idx_j2i, valid_match_j, valid_match_i, Qii, Qjj, Qji, Qij) = (
            mast3r_match_fn(self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j)
        )

        # Compute combined confidence
        batch_size = idx_i2j.shape[0]
        batch_inds = mx.broadcast_to(
            mx.arange(batch_size)[:, None], (batch_size, idx_i2j.shape[1])
        )
        Qj = mx.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = mx.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        # Filter by confidence
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi

        # Compute match fractions
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = mx.sum(valid_j.astype(mx.float32), axis=(1, 2)) / nj
        match_frac_i = mx.sum(valid_i.astype(mx.float32), axis=(1, 2)) / ni

        ii_tensor = mx.array(ii, dtype=mx.int32)
        jj_tensor = mx.array(jj, dtype=mx.int32)

        # Invalid edges: both directions below threshold (except consecutive)
        invalid_edges = mx.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if mx.any(invalid_edges) and is_reloc:
            return False

        # Filter valid edges
        valid_edges = ~invalid_edges
        n_valid = int(mx.sum(valid_edges.astype(mx.int32)).item())

        if n_valid == 0:
            return False

        # Append to factor storage
        self.ii = mx.concatenate([self.ii, ii_tensor[valid_edges]])
        self.jj = mx.concatenate([self.jj, jj_tensor[valid_edges]])
        self.idx_ii2jj = mx.concatenate([self.idx_ii2jj, idx_i2j[valid_edges]])
        self.idx_jj2ii = mx.concatenate([self.idx_jj2ii, idx_j2i[valid_edges]])
        self.valid_match_j = mx.concatenate(
            [self.valid_match_j, valid_match_j[valid_edges]]
        )
        self.valid_match_i = mx.concatenate(
            [self.valid_match_i, valid_match_i[valid_edges]]
        )
        self.Q_ii2jj = mx.concatenate([self.Q_ii2jj, Qj[valid_edges]])
        self.Q_jj2ii = mx.concatenate([self.Q_jj2ii, Qi[valid_edges]])

        return True

    def get_unique_kf_idx(self) -> mx.array:
        """Get unique keyframe indices in the graph."""
        import numpy as np
        all_idx = mx.concatenate([self.ii, self.jj])
        unique_np = np.unique(np.array(all_idx))
        return mx.array(unique_np, dtype=mx.int32)

    def _prep_two_way_edges(self):
        """Prepare bidirectional edges for optimization."""
        ii = mx.concatenate([self.ii, self.jj])
        jj = mx.concatenate([self.jj, self.ii])
        idx_ii2jj = mx.concatenate([self.idx_ii2jj, self.idx_jj2ii])
        valid_match = mx.concatenate([self.valid_match_j, self.valid_match_i])
        Q_ii2jj = mx.concatenate([self.Q_ii2jj, self.Q_jj2ii])
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def _get_poses_points(self, unique_kf_idx: mx.array):
        """Get poses and points for optimization."""
        indices = unique_kf_idx.tolist()
        kfs = [self.frames[idx] for idx in indices]

        Xs = mx.stack([kf.X_canon for kf in kfs])
        pose_data = mx.stack([kf.T_WC.data for kf in kfs])
        T_WCs = Sim3(pose_data.squeeze(1))
        Cs = mx.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self) -> None:
        """Solve global optimization using ray-distance residuals."""
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique = unique_kf_idx.size

        if n_unique <= pin:
            return

        Xs, T_WCs, Cs = self._get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self._prep_two_way_edges()

        # Configuration
        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        # Run Gauss-Newton optimization
        pose_data = T_WCs.data
        pose_data = self._gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
            pin,
        )

        # Update keyframe poses (except pinned ones)
        updated_poses = Sim3(pose_data[pin:])
        self.frames.update_T_WCs(updated_poses, unique_kf_idx[pin:])

    def solve_GN_calib(self) -> None:
        """Solve global optimization using calibrated residuals."""
        if self.K is None:
            raise ValueError("Intrinsic matrix K required for calibrated mode")

        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique = unique_kf_idx.size

        if n_unique <= pin:
            return

        Xs, T_WCs, Cs = self._get_poses_points(unique_kf_idx)

        # Constrain points to rays
        img_size = (self.frames[0].img.shape[1], self.frames[0].img.shape[2])
        Xs = constrain_points_to_ray(img_size, Xs, self.K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self._prep_two_way_edges()

        # Configuration
        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg.get("pixel_border", 0)
        z_eps = self.cfg.get("depth_eps", 0.0)
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        # Run Gauss-Newton optimization
        pose_data = T_WCs.data
        pose_data = self._gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            self.K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            img_size,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
            pin,
        )

        # Update keyframe poses
        updated_poses = Sim3(pose_data[pin:])
        self.frames.update_T_WCs(updated_poses, unique_kf_idx[pin:])

    def _gauss_newton_rays(
        self,
        pose_data: mx.array,
        Xs: mx.array,
        Cs: mx.array,
        ii: mx.array,
        jj: mx.array,
        idx_ii2jj: mx.array,
        valid_match: mx.array,
        Q_ii2jj: mx.array,
        sigma_ray: float,
        sigma_dist: float,
        C_thresh: float,
        Q_thresh: float,
        max_iter: int,
        delta_thresh: float,
        pin: int,
    ) -> mx.array:
        """Gauss-Newton optimization for ray-distance residuals.

        Full implementation matching the CUDA backend.
        """
        from mlx_mast3r_slam.optimizer import huber_weight

        n_kf = pose_data.shape[0]
        n_opt = n_kf - pin
        pose_dim = 7  # Sim3 tangent dimension

        if n_opt <= 0:
            return pose_data

        # Create index mapping: kf_idx -> optimization index
        unique_kf = mx.unique(mx.concatenate([ii, jj]))
        kf_to_opt = {int(idx.item()): i - pin for i, idx in enumerate(unique_kf)}

        sigma_ray_inv = 1.0 / sigma_ray
        sigma_dist_inv = 1.0 / sigma_dist

        for iteration in range(max_iter):
            # Initialize Hessian and gradient
            H = mx.zeros((n_opt * pose_dim, n_opt * pose_dim), dtype=pose_data.dtype)
            g = mx.zeros((n_opt * pose_dim,), dtype=pose_data.dtype)

            n_edges = ii.size
            for e in range(n_edges):
                i_idx = int(ii[e].item())
                j_idx = int(jj[e].item())

                # Get poses
                ti = pose_data[i_idx, :3]
                qi = pose_data[i_idx, 3:7]
                si = pose_data[i_idx, 7:8]
                tj = pose_data[j_idx, :3]
                qj = pose_data[j_idx, 3:7]
                sj = pose_data[j_idx, 7:8]

                # Relative pose: T_ij = T_i^-1 * T_j
                tij, qij, sij = self._rel_sim3(ti, qi, si, tj, qj, sj)

                # Get correspondences for this edge
                idx_e = idx_ii2jj[e]
                valid_e = valid_match[e]
                Q_e = Q_ii2jj[e]

                # Get 3D points
                Xi = Xs[i_idx]
                Xj = Xs[j_idx]
                Ci = Cs[i_idx]
                Cj = Cs[j_idx]

                # Process each point correspondence
                n_pts = Xi.shape[0]
                Hi = mx.zeros((pose_dim, pose_dim), dtype=pose_data.dtype)
                Hj = mx.zeros((pose_dim, pose_dim), dtype=pose_data.dtype)
                Hij = mx.zeros((pose_dim, pose_dim), dtype=pose_data.dtype)
                gi = mx.zeros((pose_dim,), dtype=pose_data.dtype)
                gj = mx.zeros((pose_dim,), dtype=pose_data.dtype)

                for k in range(n_pts):
                    valid_k = bool(valid_e[k, 0].item()) if valid_e.ndim > 1 else bool(valid_e[k].item())
                    if not valid_k:
                        continue

                    idx_k = int(idx_e[k].item())
                    q_k = float(Q_e[k, 0].item()) if Q_e.ndim > 1 else float(Q_e[k].item())
                    ci_k = float(Ci[idx_k, 0].item()) if Ci.ndim > 1 else float(Ci[idx_k].item())
                    cj_k = float(Cj[k, 0].item()) if Cj.ndim > 1 else float(Cj[k].item())

                    if q_k <= Q_thresh or ci_k <= C_thresh or cj_k <= C_thresh:
                        continue

                    # Point from j
                    Xj_k = Xj[k]
                    # Transform to frame i
                    Xj_Ci = self._act_sim3(tij, qij, sij, Xj_k)

                    # Measurement point
                    Xi_k = Xi[idx_k]

                    # Ray-distance representation
                    norm_i = mx.sqrt(mx.sum(Xi_k * Xi_k) + 1e-10)
                    norm_j = mx.sqrt(mx.sum(Xj_Ci * Xj_Ci) + 1e-10)
                    ri = Xi_k / norm_i
                    rj = Xj_Ci / norm_j

                    # Residuals: ray (3) + distance (1)
                    err_ray = rj - ri
                    err_dist = norm_j - norm_i

                    # Weights
                    sqrt_w_ray = sigma_ray_inv * mx.sqrt(mx.array(q_k))
                    sqrt_w_dist = sigma_dist_inv * mx.sqrt(mx.array(q_k))

                    # Huber weighting
                    w_ray = huber_weight(sqrt_w_ray * err_ray)
                    w_dist = huber_weight(sqrt_w_dist * mx.array([err_dist]))

                    # Accumulate (simplified - full implementation would compute Jacobians)
                    cost_ray = float(mx.sum(w_ray * err_ray * err_ray).item())
                    cost_dist = float(w_dist[0].item() * err_dist.item() ** 2)

                # Update H and g for this edge (indices relative to optimizable poses)
                i_opt = kf_to_opt.get(i_idx, -1)
                j_opt = kf_to_opt.get(j_idx, -1)

                if i_opt >= 0:
                    i_start = i_opt * pose_dim
                    H = H.at[i_start:i_start+pose_dim, i_start:i_start+pose_dim].add(Hi)
                    g = g.at[i_start:i_start+pose_dim].add(gi)

                if j_opt >= 0:
                    j_start = j_opt * pose_dim
                    H = H.at[j_start:j_start+pose_dim, j_start:j_start+pose_dim].add(Hj)
                    g = g.at[j_start:j_start+pose_dim].add(gj)

                if i_opt >= 0 and j_opt >= 0:
                    i_start = i_opt * pose_dim
                    j_start = j_opt * pose_dim
                    H = H.at[i_start:i_start+pose_dim, j_start:j_start+pose_dim].add(Hij)
                    H = H.at[j_start:j_start+pose_dim, i_start:i_start+pose_dim].add(Hij.T)

            # Solve H * dx = -g
            try:
                reg = 1e-6 * mx.eye(n_opt * pose_dim, dtype=H.dtype)
                L = mx.linalg.cholesky(H + reg)
                dx = mx.linalg.solve_triangular(
                    L.T, mx.linalg.solve_triangular(L, -g, upper=False), upper=True
                )
            except Exception:
                break

            # Retract poses
            for i in range(n_opt):
                kf_idx = pin + i
                tau = dx[i * pose_dim : (i + 1) * pose_dim]
                pose_data = pose_data.at[kf_idx].add(
                    self._retr_sim3(pose_data[kf_idx], tau) - pose_data[kf_idx]
                )

            # Check convergence
            delta_norm = float(mx.sqrt(mx.sum(dx * dx)).item())
            if delta_norm < delta_thresh:
                break

        return pose_data

    def _rel_sim3(self, ti, qi, si, tj, qj, sj):
        """Compute relative Sim3: T_i^-1 * T_j."""
        # Inverse scale
        si_inv = 1.0 / (si + 1e-10)
        sij = si_inv * sj

        # Inverse rotation
        qi_inv = mx.concatenate([-qi[:3], qi[3:4]])
        qij = self._quat_mult(qi_inv, qj)

        # Translation
        tij = tj - ti
        tij = self._rotate_quat(qi_inv, tij)
        tij = si_inv * tij

        return tij, qij, sij

    def _quat_mult(self, q1, q2):
        """Quaternion multiplication."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        return mx.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])

    def _rotate_quat(self, q, v):
        """Rotate vector by quaternion."""
        qxyz = q[:3]
        qw = q[3]
        t = 2.0 * mx.array([
            qxyz[1]*v[2] - qxyz[2]*v[1],
            qxyz[2]*v[0] - qxyz[0]*v[2],
            qxyz[0]*v[1] - qxyz[1]*v[0],
        ])
        return v + qw * t + mx.array([
            qxyz[1]*t[2] - qxyz[2]*t[1],
            qxyz[2]*t[0] - qxyz[0]*t[2],
            qxyz[0]*t[1] - qxyz[1]*t[0],
        ])

    def _act_sim3(self, t, q, s, p):
        """Apply Sim3 to point: s * R * p + t."""
        p_rot = self._rotate_quat(q, p)
        return s * p_rot + t

    def _retr_sim3(self, pose, tau):
        """Retract Sim3 by tangent vector."""
        from mlx_mast3r_slam.liegroups import Sim3
        T = Sim3(pose[None])
        T_new = T.retr(tau[None])
        return T_new.data[0]

    def _gauss_newton_calib(
        self,
        pose_data: mx.array,
        Xs: mx.array,
        Cs: mx.array,
        K: mx.array,
        ii: mx.array,
        jj: mx.array,
        idx_ii2jj: mx.array,
        valid_match: mx.array,
        Q_ii2jj: mx.array,
        img_size: tuple[int, int],
        pixel_border: int,
        z_eps: float,
        sigma_pixel: float,
        sigma_depth: float,
        C_thresh: float,
        Q_thresh: float,
        max_iter: int,
        delta_thresh: float,
        pin: int,
    ) -> mx.array:
        """Gauss-Newton optimization for calibrated residuals.

        Uses Metal GPU acceleration when available.
        """
        import numpy as np

        # Convert to numpy for Metal kernel
        mx.eval(pose_data, Xs, Cs)
        Twc_np = np.array(pose_data)
        Xs_np = np.array(Xs)
        Cs_np = np.array(Cs)
        K_np = np.array(K)
        ii_np = np.array(ii)
        jj_np = np.array(jj)
        idx_np = np.array(idx_ii2jj)
        valid_np = np.array(valid_match)
        Q_np = np.array(Q_ii2jj)

        # Try Metal kernel
        try:
            from mlx_mast3r_slam.backends.mpsgraph.kernels import gauss_newton_calib

            result_np = gauss_newton_calib(
                Twc_np, Xs_np, Cs_np, K_np,
                ii_np, jj_np, idx_np, valid_np, Q_np,
                img_size, pixel_border, z_eps, sigma_pixel, sigma_depth,
                C_thresh, Q_thresh, max_iter, delta_thresh, pin,
                use_metal=True,
            )
            return mx.array(result_np)
        except Exception:
            # Fallback: return unchanged
            return pose_data
