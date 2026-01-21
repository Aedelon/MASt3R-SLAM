# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Metal shader runner for SLAM kernels.

Compiles and runs custom Metal shaders for GPU-accelerated operations.
"""

from __future__ import annotations

import os
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


class MetalRunner:
    """Runs custom Metal compute shaders."""

    def __init__(self):
        if not _METAL_AVAILABLE:
            raise RuntimeError(
                "Metal not available. Install PyObjC: pip install pyobjc-framework-Metal"
            )

        self.device = _mtl.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        self.command_queue = self.device.newCommandQueue()
        self.library: Optional[object] = None
        self.pipelines: dict = {}

        # Load shaders
        self._load_shaders()

    def _load_shaders(self):
        """Load and compile Metal shaders."""
        shader_dir = Path(__file__).parent / "shaders"

        # Compile all .metal files
        for metal_file in shader_dir.glob("*.metal"):
            self._compile_shader(metal_file)

    def _compile_shader(self, shader_path: Path):
        """Compile a Metal shader file."""
        with open(shader_path, "r") as f:
            source = f.read()

        # Compile shader
        options = _mtl.MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(source, options, None)

        if error:
            raise RuntimeError(f"Shader compilation error: {error}")

        self.library = library

        # Create pipeline for each kernel function
        for name in ["iter_proj_kernel", "validate_matches_kernel"]:
            func = library.newFunctionWithName_(name)
            if func:
                pipeline, error = self.device.newComputePipelineStateWithFunction_error_(func, None)
                if error:
                    print(f"Warning: Could not create pipeline for {name}: {error}")
                else:
                    self.pipelines[name] = pipeline

    def _create_buffer(self, data: np.ndarray) -> object:
        """Create a Metal buffer from numpy array."""
        data = np.ascontiguousarray(data)
        buffer = self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            data.nbytes,
            _mtl.MTLResourceStorageModeShared,
        )
        return buffer

    def _create_buffer_empty(self, size: int, dtype=np.float32) -> object:
        """Create an empty Metal buffer."""
        nbytes = size * np.dtype(dtype).itemsize
        buffer = self.device.newBufferWithLength_options_(
            nbytes,
            _mtl.MTLResourceStorageModeShared,
        )
        return buffer

    def _read_buffer(self, buffer: object, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Read data from Metal buffer to numpy array."""
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize

        # Use as_buffer() to get a Python buffer from Metal buffer contents
        # This is the correct PyObjC method for MTLBuffer
        buf = buffer.contents().as_buffer(nbytes)
        data = np.frombuffer(buf, dtype=dtype, count=size).reshape(shape)
        return data.copy()

    def iter_proj(
        self,
        rays_with_grad: np.ndarray,
        pts3d_norm: np.ndarray,
        p_init: np.ndarray,
        max_iter: int = 10,
        lambda_init: float = 1e-8,
        convergence_thresh: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run iterative projection kernel on GPU.

        Args:
            rays_with_grad: [B, H, W, 9] ray image with gradients
            pts3d_norm: [B, N, 3] normalized target points
            p_init: [B, N, 2] initial pixel positions
            max_iter: maximum LM iterations
            lambda_init: initial damping
            convergence_thresh: convergence threshold

        Returns:
            p_out: [B, N, 2] refined positions
            valid: [B, N] validity mask
        """
        pipeline = self.pipelines.get("iter_proj_kernel")
        if pipeline is None:
            raise RuntimeError("iter_proj_kernel not compiled")

        B, H, W, _ = rays_with_grad.shape
        N = pts3d_norm.shape[1]

        # Create buffers
        rays_buf = self._create_buffer(rays_with_grad.astype(np.float32))
        pts_buf = self._create_buffer(pts3d_norm.astype(np.float32))
        p_init_buf = self._create_buffer(p_init.astype(np.float32))
        p_out_buf = self._create_buffer_empty(B * N * 2, np.float32)
        valid_buf = self._create_buffer_empty(B * N, np.uint8)  # bool as uint8

        # Constants
        B_buf = self._create_buffer(np.array([B], dtype=np.int32))
        H_buf = self._create_buffer(np.array([H], dtype=np.int32))
        W_buf = self._create_buffer(np.array([W], dtype=np.int32))
        N_buf = self._create_buffer(np.array([N], dtype=np.int32))
        max_iter_buf = self._create_buffer(np.array([max_iter], dtype=np.int32))
        lambda_buf = self._create_buffer(np.array([lambda_init], dtype=np.float32))
        thresh_buf = self._create_buffer(np.array([convergence_thresh], dtype=np.float32))

        # Create command buffer
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(rays_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(pts_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(p_init_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(p_out_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(valid_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(B_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(H_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(W_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(N_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(max_iter_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(lambda_buf, 0, 10)
        encoder.setBuffer_offset_atIndex_(thresh_buf, 0, 11)

        # Dispatch threads
        total_threads = B * N
        threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())

        # Use dispatchThreads for direct thread count (not threadgroups)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            _mtl.MTLSizeMake(total_threads, 1, 1),
            _mtl.MTLSizeMake(threads_per_group, 1, 1),
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read results
        p_out = self._read_buffer(p_out_buf, (B, N, 2), np.float32)
        valid = self._read_buffer(valid_buf, (B, N), np.uint8).astype(bool)

        return p_out, valid


# Global instance
_runner: Optional[MetalRunner] = None


def get_runner() -> MetalRunner:
    """Get or create the global MetalRunner instance."""
    global _runner
    if _runner is None:
        _runner = MetalRunner()
    return _runner


def iter_proj_metal(
    rays_with_grad: np.ndarray,
    pts3d_norm: np.ndarray,
    p_init: np.ndarray,
    max_iter: int = 10,
    lambda_init: float = 1e-8,
    convergence_thresh: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run iterative projection using Metal GPU acceleration."""
    runner = get_runner()
    return runner.iter_proj(
        rays_with_grad, pts3d_norm, p_init, max_iter, lambda_init, convergence_thresh
    )
