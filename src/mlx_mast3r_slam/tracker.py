# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Frame tracker (frontend) for MLX-MASt3R-SLAM."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_mast3r_slam.frame import Frame, Keyframes
from mlx_mast3r_slam.liegroups import Sim3
from mlx_mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mlx_mast3r_slam.optimizer import check_convergence, huber_weight
from mlx_mast3r_slam.config import get_config


class FrameTracker:
    """Frame tracker for visual odometry.

    Tracks camera pose relative to the last keyframe using
    MASt3R feature matching and Gauss-Newton optimization.
    """

    def __init__(
        self,
        model,  # MASt3R model
        keyframes: Keyframes,
    ) -> None:
        """Initialize tracker.

        Args:
            model: MASt3R model for feature extraction
            keyframes: Shared keyframe storage
        """
        self.model = model
        self.keyframes = keyframes
        self.cfg = get_config()["tracking"]

        self.idx_f2k: Optional[mx.array] = None

    def reset_idx_f2k(self) -> None:
        """Reset correspondence indices."""
        self.idx_f2k = None

    def track(
        self,
        frame: Frame,
        mast3r_match_fn,
    ) -> tuple[bool, list, bool]:
        """Track frame against last keyframe.

        Args:
            frame: Current frame to track
            mast3r_match_fn: Function for MASt3R matching

        Returns:
            new_kf: Whether to create new keyframe
            match_info: Matching information for visualization
            try_reloc: Whether to try relocalization
        """
        keyframe = self.keyframes.last_keyframe()
        if keyframe is None:
            return False, [], True

        # Match frame to keyframe
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_fn(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )

        # Save indices for next frame
        self.idx_f2k = idx_f2k

        # Remove batch dimension
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        # Force evaluation to free memory
        mx.eval(idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf)

        # Compute combined confidence - handle shape properly
        # Qff: [1, H*W, 1], idx_f2k: [H*W], Qkf: [1, H*W, 1]
        Qff_squeezed = Qff[0, :, 0]  # [H*W]
        Qkf_squeezed = Qkf[0, :, 0]  # [H*W]
        Qk = mx.sqrt(Qff_squeezed[idx_f2k] * Qkf_squeezed)  # [H*W]
        Qk = Qk[:, None]  # [H*W, 1]

        # Update frame pointmap
        frame.update_pointmap(Xff, Cff)

        # Get configuration
        use_calib = self.cfg.get("use_calib", False)
        img_size = (frame.img.shape[1], frame.img.shape[2])

        K = keyframe.K if use_calib else None

        # Get points and poses
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self._get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Validity masks
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        # Check minimum matches
        match_frac = mx.sum(valid_opt.astype(mx.float32)) / valid_opt.size
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            if not use_calib:
                T_WCf, T_CkCf = self._opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self._opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Optimization failed for frame {frame.frame_id}: {e}")
            return False, [], True

        frame.T_WC = T_WCf

        # Update keyframe pointmap with transformed points
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = mx.sum(valid_kf.astype(mx.float32))
        match_frac_k = n_valid / valid_kf.size

        # Count unique matches
        unique_idx = mx.unique(idx_f2k[valid_match_k[:, 0]])
        unique_frac_f = unique_idx.shape[0] / valid_kf.size

        new_kf = (
            min(float(match_frac_k.item()), float(unique_frac_f))
            < self.cfg["match_frac_thresh"]
        )

        if new_kf:
            self.reset_idx_f2k()

        match_info = [
            keyframe.X_canon,
            keyframe.get_average_conf(),
            frame.X_canon,
            frame.get_average_conf(),
            Qkf,
            Qff,
        ]

        return new_kf, match_info, False

    def _get_points_poses(
        self,
        frame: Frame,
        keyframe: Frame,
        idx_f2k: mx.array,
        img_size: tuple[int, int],
        use_calib: bool,
        K: Optional[mx.array],
    ):
        """Get points and poses for optimization."""
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib and K is not None:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, dtype=Xf.dtype)
            uv_k = uv_k.reshape(-1, 2)
            meas_k = mx.concatenate([uv_k, mx.log(Xk[..., 2:3] + 1e-10)], axis=-1)

            valid_meas_k = Xk[..., 2:3] > self.cfg.get("depth_eps", 0.0)
            meas_k = mx.where(
                mx.broadcast_to(valid_meas_k, meas_k.shape),
                meas_k,
                mx.zeros_like(meas_k),
            )

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def _solve(
        self,
        sqrt_info: mx.array,
        r: mx.array,
        J: mx.array,
    ) -> tuple[mx.array, float]:
        """Solve one Gauss-Newton step.

        Args:
            sqrt_info: Square root information weights
            r: Residuals
            J: Jacobian

        Returns:
            delta: Update vector
            cost: Current cost
        """
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * mx.sqrt(
            huber_weight(whitened_r, k=self.cfg["huber"])
        )

        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).reshape(-1, mdim)
        b = (robust_sqrt_info * r).reshape(-1, 1)

        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * float((b.T @ b).item())

        # Solve using optimized backend (numpy with Accelerate on macOS)
        import numpy as np
        from mlx_mast3r_slam.backends.mpsgraph.linalg import cholesky_solve

        H_np = np.array(H)
        g_np = np.array(g).squeeze()

        tau_np = cholesky_solve(H_np, g_np, reg=1e-6)
        tau = mx.array(tau_np)

        return tau.reshape(1, -1), cost

    def _opt_pose_ray_dist_sim3(
        self,
        Xf: mx.array,
        Xk: mx.array,
        T_WCf: Sim3,
        T_WCk: Sim3,
        Qk: mx.array,
        valid: mx.array,
    ) -> tuple[Sim3, Sim3]:
        """Optimize pose using ray-distance residuals.

        Args:
            Xf: Frame points
            Xk: Keyframe points
            T_WCf: Frame pose
            T_WCk: Keyframe pose
            Qk: Match confidence
            valid: Validity mask

        Returns:
            T_WCf: Optimized frame pose
            T_CkCf: Relative pose
        """
        sqrt_info_ray = 1.0 / self.cfg["sigma_ray"] * valid * mx.sqrt(Qk)
        sqrt_info_dist = 1.0 / self.cfg["sigma_dist"] * valid * mx.sqrt(Qk)
        sqrt_info = mx.concatenate(
            [
                mx.broadcast_to(sqrt_info_ray, sqrt_info_ray.shape[:-1] + (3,)),
                sqrt_info_dist,
            ],
            axis=-1,
        )

        # Relative pose
        T_CkCf = T_WCk.inv() * T_WCf

        # Pre-compute ray-distance for keyframe points
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # Transform frame points to keyframe frame
            Xf_Ck, dXf_Ck_dT = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_dX = point_to_ray_dist(Xf_Ck, jacobian=True)

            # Residual: r = z - h(x)
            r = rd_k - rd_f_Ck

            # Jacobian: chain rule
            J = -drd_f_dX @ dXf_Ck_dT

            tau, new_cost = self._solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau,
            ):
                break
            old_cost = new_cost

        T_WCf = T_WCk * T_CkCf
        return T_WCf, T_CkCf

    def _opt_pose_calib_sim3(
        self,
        Xf: mx.array,
        Xk: mx.array,
        T_WCf: Sim3,
        T_WCk: Sim3,
        Qk: mx.array,
        valid: mx.array,
        meas_k: mx.array,
        valid_meas_k: mx.array,
        K: mx.array,
        img_size: tuple[int, int],
    ) -> tuple[Sim3, Sim3]:
        """Optimize pose using calibrated projection residuals.

        Args:
            Xf: Frame points
            Xk: Keyframe points
            T_WCf: Frame pose
            T_WCk: Keyframe pose
            Qk: Match confidence
            valid: Validity mask
            meas_k: Keyframe measurements [u, v, log(z)]
            valid_meas_k: Valid measurement mask
            K: Intrinsic matrix
            img_size: Image size

        Returns:
            T_WCf: Optimized frame pose
            T_CkCf: Relative pose
        """
        sqrt_info_pixel = 1.0 / self.cfg["sigma_pixel"] * valid * mx.sqrt(Qk)
        sqrt_info_depth = 1.0 / self.cfg["sigma_depth"] * valid * mx.sqrt(Qk)
        sqrt_info = mx.concatenate(
            [
                mx.broadcast_to(sqrt_info_pixel, sqrt_info_pixel.shape[:-1] + (2,)),
                sqrt_info_depth,
            ],
            axis=-1,
        )

        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_dX, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg.get("pixel_border", 0),
                z_eps=self.cfg.get("depth_eps", 0.0),
            )

            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = mx.where(
                mx.broadcast_to(valid2, sqrt_info.shape),
                sqrt_info,
                mx.zeros_like(sqrt_info),
            )

            r = meas_k - pzf_Ck
            J = -dpzf_dX @ dXf_Ck_dT

            tau, new_cost = self._solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau,
            ):
                break
            old_cost = new_cost

        T_WCf = T_WCk * T_CkCf
        return T_WCf, T_CkCf
