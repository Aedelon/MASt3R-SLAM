# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Metal GPU runner for Gauss-Newton optimization."""

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


class GNMetalRunner:
    """Runs Gauss-Newton optimization using Metal GPU acceleration."""

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
        shader_path = Path(__file__).parent / "shaders" / "gauss_newton.metal"

        with open(shader_path, "r") as f:
            source = f.read()

        options = _mtl.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)

        if error:
            raise RuntimeError(f"Shader compilation error: {error}")

        # Create pipelines
        for name in ["gn_jacobian_kernel", "pose_update_kernel"]:
            func = library.newFunctionWithName_(name)
            if func:
                pipeline, err = self.device.newComputePipelineStateWithFunction_error_(func, None)
                if err:
                    print(f"Warning: Could not create pipeline for {name}: {err}")
                else:
                    self.pipelines[name] = pipeline

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

    def gauss_newton_rays(
        self,
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
        """Run Gauss-Newton optimization with Metal GPU acceleration."""
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

        # Prepare data
        Twc_work = Twc.astype(np.float32).copy()
        Xs_flat = Xs.astype(np.float32)
        Cs_flat = Cs.astype(np.float32)

        sigma_inv = np.float32(1.0 / sigma_ray)

        # Create input buffers
        Twc_buf = self._create_buffer(Twc_work)
        Xs_buf = self._create_buffer(Xs_flat)
        Cs_buf = self._create_buffer(Cs_flat)
        ii_buf = self._create_buffer(ii.astype(np.int32))
        jj_buf = self._create_buffer(jj.astype(np.int32))
        idx_buf = self._create_buffer(idx_ii2jj.astype(np.int32).flatten())
        valid_buf = self._create_buffer(valid_match.astype(np.uint8).flatten())
        Q_buf = self._create_buffer(Q.astype(np.float32).flatten())

        # Constants
        num_kf_buf = self._create_buffer(np.array([num_kf], dtype=np.int32))
        num_pts_buf = self._create_buffer(np.array([num_pts], dtype=np.int32))
        num_edges_buf = self._create_buffer(np.array([num_edges], dtype=np.int32))
        sigma_inv_buf = self._create_buffer(np.array([sigma_inv], dtype=np.float32))
        C_thresh_buf = self._create_buffer(np.array([C_thresh], dtype=np.float32))
        Q_thresh_buf = self._create_buffer(np.array([Q_thresh], dtype=np.float32))

        free_kf_idx = unique_kf[pin:].astype(np.int32)
        free_kf_buf = self._create_buffer(free_kf_idx)
        num_free_buf = self._create_buffer(np.array([num_free], dtype=np.int32))

        jac_pipeline = self.pipelines.get("gn_jacobian_kernel")
        update_pipeline = self.pipelines.get("pose_update_kernel")

        if jac_pipeline is None:
            raise RuntimeError("gn_jacobian_kernel not compiled")

        # Upper triangular indices for 7x7 matrix (28 values)
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
            encoder.setBuffer_offset_atIndex_(JtJ_i_buf, 0, 8)
            encoder.setBuffer_offset_atIndex_(Jtr_i_buf, 0, 9)
            encoder.setBuffer_offset_atIndex_(JtJ_j_buf, 0, 10)
            encoder.setBuffer_offset_atIndex_(Jtr_j_buf, 0, 11)
            encoder.setBuffer_offset_atIndex_(JtJ_ij_buf, 0, 12)
            encoder.setBuffer_offset_atIndex_(valid_out_buf, 0, 13)
            encoder.setBuffer_offset_atIndex_(num_kf_buf, 0, 14)
            encoder.setBuffer_offset_atIndex_(num_pts_buf, 0, 15)
            encoder.setBuffer_offset_atIndex_(num_edges_buf, 0, 16)
            encoder.setBuffer_offset_atIndex_(sigma_inv_buf, 0, 17)
            encoder.setBuffer_offset_atIndex_(C_thresh_buf, 0, 18)
            encoder.setBuffer_offset_atIndex_(Q_thresh_buf, 0, 19)

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

            # Pre-compute edge indices and local indices for all points
            edge_indices = np.arange(total_pts) // num_pts
            ix_all = ii[edge_indices]
            jx_all = jj[edge_indices]

            # Build local index arrays
            i_local_all = np.array([kf_to_local.get(int(ix), -pin - 1) for ix in ix_all])
            j_local_all = np.array([kf_to_local.get(int(jx), -pin - 1) for jx in jx_all])

            # Expand upper triangular indices for vectorized operation
            triu_row = triu_idx[:, 0]
            triu_col = triu_idx[:, 1]

            # Process valid points only
            valid_mask = valid_out
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                # Get valid data
                i_local_valid = i_local_all[valid_indices]
                j_local_valid = j_local_all[valid_indices]
                JtJ_i_valid = JtJ_i[valid_indices]
                JtJ_j_valid = JtJ_j[valid_indices]
                JtJ_ij_valid = JtJ_ij[valid_indices]
                Jtr_i_valid = Jtr_i[valid_indices]
                Jtr_j_valid = Jtr_j[valid_indices]

                # Accumulate per keyframe using numpy operations
                for kf_local in range(num_free):
                    # Points where i == kf
                    mask_i = i_local_valid == kf_local
                    if np.any(mask_i):
                        # Sum JtJ_i contributions
                        JtJ_sum = JtJ_i_valid[mask_i].sum(axis=0)
                        # Expand to 7x7
                        for idx, (m, n) in enumerate(triu_idx):
                            H[kf_local * 7 + m, kf_local * 7 + n] += JtJ_sum[idx]
                            if m != n:
                                H[kf_local * 7 + n, kf_local * 7 + m] += JtJ_sum[idx]
                        # Sum gradient
                        g[kf_local * 7 : (kf_local + 1) * 7] += Jtr_i_valid[mask_i].sum(axis=0)

                    # Points where j == kf
                    mask_j = j_local_valid == kf_local
                    if np.any(mask_j):
                        JtJ_sum = JtJ_j_valid[mask_j].sum(axis=0)
                        for idx, (m, n) in enumerate(triu_idx):
                            H[kf_local * 7 + m, kf_local * 7 + n] += JtJ_sum[idx]
                            if m != n:
                                H[kf_local * 7 + n, kf_local * 7 + m] += JtJ_sum[idx]
                        g[kf_local * 7 : (kf_local + 1) * 7] += Jtr_j_valid[mask_j].sum(axis=0)

                # Off-diagonal blocks (ij cross terms)
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

            # Update poses on GPU
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
_gn_runner: Optional[GNMetalRunner] = None


def get_gn_runner() -> GNMetalRunner:
    """Get or create global GN Metal runner."""
    global _gn_runner
    if _gn_runner is None:
        _gn_runner = GNMetalRunner()
    return _gn_runner
