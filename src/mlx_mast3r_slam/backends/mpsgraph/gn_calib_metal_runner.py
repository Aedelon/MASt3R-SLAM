# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Metal GPU runner for calibrated Gauss-Newton optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Check if Metal is available
_METAL_AVAILABLE = False
_mtl = None

try:
    import Metal as mtl

    _mtl = mtl
    _METAL_AVAILABLE = True
except ImportError:
    pass


class GNCalibMetalRunner:
    """Runs calibrated Gauss-Newton optimization using Metal GPU acceleration."""

    def __init__(self):
        if not _METAL_AVAILABLE:
            raise RuntimeError("Metal not available")

        self.device = _mtl.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        self.command_queue = self.device.newCommandQueue()
        self.pipelines = {}

        self._load_shaders()

    def _load_shaders(self):
        """Load and compile Metal shaders."""
        # Load calib kernel
        shader_path = Path(__file__).parent / "shaders" / "gauss_newton_calib.metal"
        with open(shader_path, "r") as f:
            source = f.read()

        options = _mtl.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)

        if error:
            raise RuntimeError(f"Shader compilation error: {error}")

        # Create pipeline for calib kernel
        func = library.newFunctionWithName_("gn_calib_jacobian_kernel")
        if func:
            pipeline, err = self.device.newComputePipelineStateWithFunction_error_(func, None)
            if err:
                raise RuntimeError(f"Could not create pipeline: {err}")
            self.pipelines["gn_calib_jacobian_kernel"] = pipeline

        # Load pose update kernel from gauss_newton.metal
        gn_shader_path = Path(__file__).parent / "shaders" / "gauss_newton.metal"
        with open(gn_shader_path, "r") as f:
            gn_source = f.read()

        gn_library, gn_error = self.device.newLibraryWithSource_options_error_(
            gn_source, options, None
        )
        if not gn_error:
            func = gn_library.newFunctionWithName_("pose_update_kernel")
            if func:
                pipeline, _ = self.device.newComputePipelineStateWithFunction_error_(func, None)
                if pipeline:
                    self.pipelines["pose_update_kernel"] = pipeline

    def _create_buffer(self, data: np.ndarray):
        """Create Metal buffer from numpy array."""
        data = np.ascontiguousarray(data)
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(), data.nbytes, _mtl.MTLResourceStorageModeShared
        )

    def _create_buffer_empty(self, nbytes: int):
        """Create empty Metal buffer."""
        return self.device.newBufferWithLength_options_(nbytes, _mtl.MTLResourceStorageModeShared)

    def _read_buffer(self, buffer, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Read Metal buffer to numpy array."""
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize
        buf = buffer.contents().as_buffer(nbytes)
        return np.frombuffer(buf, dtype=dtype, count=size).reshape(shape).copy()

    def gauss_newton_calib(
        self,
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
        """Run calibrated Gauss-Newton optimization with Metal GPU acceleration.

        Args:
            Twc: Poses [num_kf, 8]
            Xs: 3D points [num_kf, num_pts, 3]
            Cs: Confidences [num_kf, num_pts]
            K: Intrinsic matrix [3, 3] or [4] = (fx, fy, cx, cy)
            ii, jj: Edge indices
            idx_ii2jj: Point correspondences
            valid_match: Match validity
            Q: Match confidence
            img_size: (width, height)
            pixel_border: Border to exclude
            z_eps: Minimum depth
            sigma_pixel: Pixel residual weight
            sigma_depth: Depth residual weight
            C_thresh, Q_thresh: Confidence thresholds
            max_iter: Maximum iterations
            delta_thresh: Convergence threshold
            pin: Number of fixed poses

        Returns:
            Updated Twc poses
        """
        num_kf = Twc.shape[0]
        num_edges = len(ii)
        num_pts = Xs.shape[1]
        total_pts = num_edges * num_pts

        if num_edges == 0 or num_kf <= pin:
            return Twc.copy()

        # Get unique keyframes
        unique_kf = np.unique(np.concatenate([ii, jj]))
        num_unique = len(unique_kf)

        if num_unique <= pin:
            return Twc.copy()

        kf_to_local = {int(kf): i - pin for i, kf in enumerate(unique_kf)}
        num_free = num_unique - pin

        if num_free <= 0:
            return Twc.copy()

        # Normalize shapes
        if valid_match.ndim == 3:
            valid_match = valid_match[..., 0]
        if Q.ndim == 3:
            Q = Q[..., 0]
        if Cs.ndim == 3:
            Cs = Cs[..., 0]

        # Extract intrinsics
        if K.shape == (3, 3):
            K_vec = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
        else:
            K_vec = K.astype(np.float32).flatten()[:4]

        img_width, img_height = img_size
        sigma_pixel_inv = np.float32(1.0 / sigma_pixel)
        sigma_depth_inv = np.float32(1.0 / sigma_depth)

        # Prepare data
        Twc_work = Twc.astype(np.float32).copy()
        Xs_flat = Xs.astype(np.float32)
        Cs_flat = Cs.astype(np.float32)

        # Create input buffers
        Twc_buf = self._create_buffer(Twc_work)
        Xs_buf = self._create_buffer(Xs_flat)
        Cs_buf = self._create_buffer(Cs_flat)
        ii_buf = self._create_buffer(ii.astype(np.int32))
        jj_buf = self._create_buffer(jj.astype(np.int32))
        idx_buf = self._create_buffer(idx_ii2jj.astype(np.int32).flatten())
        valid_buf = self._create_buffer(valid_match.astype(np.uint8).flatten())
        Q_buf = self._create_buffer(Q.astype(np.float32).flatten())
        K_buf = self._create_buffer(K_vec)

        # Constants
        num_kf_buf = self._create_buffer(np.array([num_kf], dtype=np.int32))
        num_pts_buf = self._create_buffer(np.array([num_pts], dtype=np.int32))
        num_edges_buf = self._create_buffer(np.array([num_edges], dtype=np.int32))
        img_width_buf = self._create_buffer(np.array([img_width], dtype=np.int32))
        img_height_buf = self._create_buffer(np.array([img_height], dtype=np.int32))
        pixel_border_buf = self._create_buffer(np.array([pixel_border], dtype=np.int32))
        z_eps_buf = self._create_buffer(np.array([z_eps], dtype=np.float32))
        sigma_pixel_inv_buf = self._create_buffer(np.array([sigma_pixel_inv], dtype=np.float32))
        sigma_depth_inv_buf = self._create_buffer(np.array([sigma_depth_inv], dtype=np.float32))
        C_thresh_buf = self._create_buffer(np.array([C_thresh], dtype=np.float32))
        Q_thresh_buf = self._create_buffer(np.array([Q_thresh], dtype=np.float32))

        free_kf_idx = unique_kf[pin:].astype(np.int32)
        free_kf_buf = self._create_buffer(free_kf_idx)
        num_free_buf = self._create_buffer(np.array([num_free], dtype=np.int32))

        jac_pipeline = self.pipelines.get("gn_calib_jacobian_kernel")
        update_pipeline = self.pipelines.get("pose_update_kernel")

        if jac_pipeline is None:
            raise RuntimeError("gn_calib_jacobian_kernel not compiled")

        # Upper triangular indices for 7x7 matrix
        triu_idx = np.array([(i, j) for i in range(7) for j in range(i, 7)])

        # Optimization loop
        for iteration in range(max_iter):
            # Allocate output buffers
            JtJ_i_buf = self._create_buffer_empty(total_pts * 28 * 4)
            Jtr_i_buf = self._create_buffer_empty(total_pts * 7 * 4)
            JtJ_j_buf = self._create_buffer_empty(total_pts * 28 * 4)
            Jtr_j_buf = self._create_buffer_empty(total_pts * 7 * 4)
            JtJ_ij_buf = self._create_buffer_empty(total_pts * 49 * 4)
            valid_out_buf = self._create_buffer_empty(total_pts)

            # Run Jacobian kernel
            cmd_buffer = self.command_queue.commandBuffer()
            encoder = cmd_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(jac_pipeline)

            encoder.setBuffer_offset_atIndex_(Twc_buf, 0, 0)
            encoder.setBuffer_offset_atIndex_(Xs_buf, 0, 1)
            encoder.setBuffer_offset_atIndex_(Cs_buf, 0, 2)
            encoder.setBuffer_offset_atIndex_(ii_buf, 0, 3)
            encoder.setBuffer_offset_atIndex_(jj_buf, 0, 4)
            encoder.setBuffer_offset_atIndex_(idx_buf, 0, 5)
            encoder.setBuffer_offset_atIndex_(valid_buf, 0, 6)
            encoder.setBuffer_offset_atIndex_(Q_buf, 0, 7)
            encoder.setBuffer_offset_atIndex_(K_buf, 0, 8)
            encoder.setBuffer_offset_atIndex_(JtJ_i_buf, 0, 9)
            encoder.setBuffer_offset_atIndex_(Jtr_i_buf, 0, 10)
            encoder.setBuffer_offset_atIndex_(JtJ_j_buf, 0, 11)
            encoder.setBuffer_offset_atIndex_(Jtr_j_buf, 0, 12)
            encoder.setBuffer_offset_atIndex_(JtJ_ij_buf, 0, 13)
            encoder.setBuffer_offset_atIndex_(valid_out_buf, 0, 14)
            encoder.setBuffer_offset_atIndex_(num_kf_buf, 0, 15)
            encoder.setBuffer_offset_atIndex_(num_pts_buf, 0, 16)
            encoder.setBuffer_offset_atIndex_(num_edges_buf, 0, 17)
            encoder.setBuffer_offset_atIndex_(img_width_buf, 0, 18)
            encoder.setBuffer_offset_atIndex_(img_height_buf, 0, 19)
            encoder.setBuffer_offset_atIndex_(pixel_border_buf, 0, 20)
            encoder.setBuffer_offset_atIndex_(z_eps_buf, 0, 21)
            encoder.setBuffer_offset_atIndex_(sigma_pixel_inv_buf, 0, 22)
            encoder.setBuffer_offset_atIndex_(sigma_depth_inv_buf, 0, 23)
            encoder.setBuffer_offset_atIndex_(C_thresh_buf, 0, 24)
            encoder.setBuffer_offset_atIndex_(Q_thresh_buf, 0, 25)

            threads_per_group = min(256, jac_pipeline.maxTotalThreadsPerThreadgroup())
            encoder.dispatchThreads_threadsPerThreadgroup_(
                _mtl.MTLSizeMake(total_pts, 1, 1),
                _mtl.MTLSizeMake(threads_per_group, 1, 1),
            )

            encoder.endEncoding()
            cmd_buffer.commit()
            cmd_buffer.waitUntilCompleted()

            # Read results
            JtJ_i = self._read_buffer(JtJ_i_buf, (total_pts, 28), np.float32)
            Jtr_i = self._read_buffer(Jtr_i_buf, (total_pts, 7), np.float32)
            JtJ_j = self._read_buffer(JtJ_j_buf, (total_pts, 28), np.float32)
            Jtr_j = self._read_buffer(Jtr_j_buf, (total_pts, 7), np.float32)
            JtJ_ij = self._read_buffer(JtJ_ij_buf, (total_pts, 49), np.float32)
            valid_out = self._read_buffer(valid_out_buf, (total_pts,), np.uint8).astype(bool)

            # Accumulate on CPU (vectorized)
            dim = 7 * num_free
            H = np.zeros((dim, dim), dtype=np.float64)
            g = np.zeros(dim, dtype=np.float64)

            edge_indices = np.arange(total_pts) // num_pts
            ix_all = ii[edge_indices]
            jx_all = jj[edge_indices]

            i_local_all = np.array([kf_to_local.get(int(ix), -pin - 1) for ix in ix_all])
            j_local_all = np.array([kf_to_local.get(int(jx), -pin - 1) for jx in jx_all])

            valid_indices = np.where(valid_out)[0]

            if len(valid_indices) > 0:
                i_local_valid = i_local_all[valid_indices]
                j_local_valid = j_local_all[valid_indices]
                JtJ_i_valid = JtJ_i[valid_indices]
                JtJ_j_valid = JtJ_j[valid_indices]
                JtJ_ij_valid = JtJ_ij[valid_indices]
                Jtr_i_valid = Jtr_i[valid_indices]
                Jtr_j_valid = Jtr_j[valid_indices]

                for kf_local in range(num_free):
                    mask_i = i_local_valid == kf_local
                    if np.any(mask_i):
                        JtJ_sum = JtJ_i_valid[mask_i].sum(axis=0)
                        for idx, (m, n) in enumerate(triu_idx):
                            H[kf_local * 7 + m, kf_local * 7 + n] += JtJ_sum[idx]
                            if m != n:
                                H[kf_local * 7 + n, kf_local * 7 + m] += JtJ_sum[idx]
                        g[kf_local * 7 : (kf_local + 1) * 7] += Jtr_i_valid[mask_i].sum(axis=0)

                    mask_j = j_local_valid == kf_local
                    if np.any(mask_j):
                        JtJ_sum = JtJ_j_valid[mask_j].sum(axis=0)
                        for idx, (m, n) in enumerate(triu_idx):
                            H[kf_local * 7 + m, kf_local * 7 + n] += JtJ_sum[idx]
                            if m != n:
                                H[kf_local * 7 + n, kf_local * 7 + m] += JtJ_sum[idx]
                        g[kf_local * 7 : (kf_local + 1) * 7] += Jtr_j_valid[mask_j].sum(axis=0)

                for kf_i in range(num_free):
                    for kf_j in range(kf_i + 1, num_free):
                        mask_ij = (i_local_valid == kf_i) & (j_local_valid == kf_j)
                        mask_ji = (i_local_valid == kf_j) & (j_local_valid == kf_i)

                        if np.any(mask_ij):
                            Hij_sum = JtJ_ij_valid[mask_ij].sum(axis=0).reshape(7, 7)
                            H[kf_i * 7 : (kf_i + 1) * 7, kf_j * 7 : (kf_j + 1) * 7] += Hij_sum
                            H[kf_j * 7 : (kf_j + 1) * 7, kf_i * 7 : (kf_i + 1) * 7] += Hij_sum.T

                        if np.any(mask_ji):
                            Hji_sum = JtJ_ij_valid[mask_ji].sum(axis=0).reshape(7, 7)
                            H[kf_j * 7 : (kf_j + 1) * 7, kf_i * 7 : (kf_i + 1) * 7] += Hji_sum
                            H[kf_i * 7 : (kf_i + 1) * 7, kf_j * 7 : (kf_j + 1) * 7] += Hji_sum.T

            # Regularization and solve
            H += np.eye(dim, dtype=np.float64) * 1e-6

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(dx) < delta_thresh:
                break

            # Update poses
            dx_buf = self._create_buffer(dx.astype(np.float32))

            if update_pipeline:
                cmd_buffer = self.command_queue.commandBuffer()
                encoder = cmd_buffer.computeCommandEncoder()
                encoder.setComputePipelineState_(update_pipeline)

                encoder.setBuffer_offset_atIndex_(Twc_buf, 0, 0)
                encoder.setBuffer_offset_atIndex_(dx_buf, 0, 1)
                encoder.setBuffer_offset_atIndex_(free_kf_buf, 0, 2)
                encoder.setBuffer_offset_atIndex_(num_free_buf, 0, 3)

                encoder.dispatchThreads_threadsPerThreadgroup_(
                    _mtl.MTLSizeMake(num_free, 1, 1),
                    _mtl.MTLSizeMake(min(64, num_free), 1, 1),
                )

                encoder.endEncoding()
                cmd_buffer.commit()
                cmd_buffer.waitUntilCompleted()

        # Read final poses
        Twc_out = self._read_buffer(Twc_buf, (num_kf, 8), np.float32)
        return Twc_out


# Global instance
_gn_calib_runner: Optional[GNCalibMetalRunner] = None


def get_gn_calib_runner() -> GNCalibMetalRunner:
    """Get or create global GN Calib Metal runner."""
    global _gn_calib_runner
    if _gn_calib_runner is None:
        _gn_calib_runner = GNCalibMetalRunner()
    return _gn_calib_runner
