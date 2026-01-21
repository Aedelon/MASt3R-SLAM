# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Metal GPU runner for match refinement."""

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


class RefineMetalRunner:
    """Runs match refinement using Metal GPU acceleration."""

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
        shader_path = Path(__file__).parent / "shaders" / "refine_matches.metal"

        with open(shader_path, "r") as f:
            source = f.read()

        options = _mtl.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)

        if error:
            raise RuntimeError(f"Shader compilation error: {error}")

        # Create pipelines
        kernel_names = [
            "refine_matches_kernel",
            "refine_matches_opt_kernel",
            "refine_matches_multiscale_kernel",
        ]
        for name in kernel_names:
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

    def refine_matches(
        self,
        D11: np.ndarray,
        D21: np.ndarray,
        p1: np.ndarray,
        radius: int = 3,
        dilation_max: int = 0,
    ) -> np.ndarray:
        """Refine matches using Metal GPU acceleration.

        Args:
            D11: Reference descriptors [B, H, W, D]
            D21: Query descriptors [B, N, D]
            p1: Initial pixel positions [B, N, 2] (x, y as integers)
            radius: Search radius
            dilation_max: Maximum dilation for multi-scale search

        Returns:
            Refined pixel positions [B, N, 2]
        """
        b, h, w, d = D11.shape
        num_pts = D21.shape[1]
        total_pts = b * num_pts

        # Use multi-scale kernel
        pipeline = self.pipelines.get("refine_matches_multiscale_kernel")
        if pipeline is None:
            pipeline = self.pipelines.get("refine_matches_kernel")
        if pipeline is None:
            raise RuntimeError("No refine_matches kernel available")

        # Prepare data
        D11_flat = D11.astype(np.float32)
        D21_flat = D21.astype(np.float32)
        p1_work = p1.astype(np.int32).copy()

        # Create buffers
        D11_buf = self._create_buffer(D11_flat)
        D21_buf = self._create_buffer(D21_flat)
        p1_buf = self._create_buffer(p1_work)

        # Constants
        batch_size_buf = self._create_buffer(np.array([b], dtype=np.int32))
        height_buf = self._create_buffer(np.array([h], dtype=np.int32))
        width_buf = self._create_buffer(np.array([w], dtype=np.int32))
        desc_dim_buf = self._create_buffer(np.array([d], dtype=np.int32))
        num_pts_buf = self._create_buffer(np.array([num_pts], dtype=np.int32))
        radius_buf = self._create_buffer(np.array([radius], dtype=np.int32))

        # Dilations from coarse to fine
        dilations = list(range(max(1, dilation_max), 0, -1))

        for dilation in dilations:
            dilation_buf = self._create_buffer(np.array([dilation], dtype=np.int32))

            cmd_buffer = self.command_queue.commandBuffer()
            encoder = cmd_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)

            encoder.setBuffer_offset_atIndex_(D11_buf, 0, 0)
            encoder.setBuffer_offset_atIndex_(D21_buf, 0, 1)
            encoder.setBuffer_offset_atIndex_(p1_buf, 0, 2)
            encoder.setBuffer_offset_atIndex_(batch_size_buf, 0, 3)
            encoder.setBuffer_offset_atIndex_(height_buf, 0, 4)
            encoder.setBuffer_offset_atIndex_(width_buf, 0, 5)
            encoder.setBuffer_offset_atIndex_(desc_dim_buf, 0, 6)
            encoder.setBuffer_offset_atIndex_(num_pts_buf, 0, 7)
            encoder.setBuffer_offset_atIndex_(radius_buf, 0, 8)
            encoder.setBuffer_offset_atIndex_(dilation_buf, 0, 9)

            threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())
            encoder.dispatchThreads_threadsPerThreadgroup_(
                _mtl.MTLSizeMake(total_pts, 1, 1),
                _mtl.MTLSizeMake(threads_per_group, 1, 1),
            )

            encoder.endEncoding()
            cmd_buffer.commit()
            cmd_buffer.waitUntilCompleted()

        # Read result
        p1_out = self._read_buffer(p1_buf, (b, num_pts, 2), np.int32)
        return p1_out


# Global instance
_refine_runner: Optional[RefineMetalRunner] = None


def get_refine_runner() -> RefineMetalRunner:
    """Get or create global refine Metal runner."""
    global _refine_runner
    if _refine_runner is None:
        _refine_runner = RefineMetalRunner()
    return _refine_runner
