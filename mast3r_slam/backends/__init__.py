# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Backend dispatcher for multi-platform compute kernels.

Auto-selects the best available backend:
1. CUDA (NVIDIA GPU) - fastest
2. CPU (OpenMP + SIMD) - portable
3. Metal (Apple Silicon) - future

Usage:
    from mast3r_slam.backends import get_backend
    backend = get_backend()
    p_new, converged = backend.iter_proj(...)
"""

from __future__ import annotations

import warnings
from typing import Optional

from mast3r_slam.backends.base import BackendBase
from mast3r_slam.device import get_device_manager

# Singleton backend instance
_backend: Optional[BackendBase] = None


def _cuda_available() -> bool:
    """Check if CUDA backend is available."""
    try:
        import mast3r_slam_backends  # noqa: F401

        return True
    except ImportError:
        return False


def _cpu_available() -> bool:
    """Check if CPU backend is available."""
    try:
        import mast3r_slam_cpu_backends  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend(force: Optional[str] = None) -> BackendBase:
    """
    Get the compute backend.

    Auto-selects based on device manager settings, or use force parameter.

    Args:
        force: Force a specific backend ("cuda", "cpu", "metal")

    Returns:
        BackendBase instance

    Raises:
        RuntimeError: If no suitable backend is available
    """
    global _backend

    # Return cached backend if available and no force override
    if _backend is not None and force is None:
        return _backend

    dm = get_device_manager()

    # Determine which backend to use
    backend_type = force
    if backend_type is None:
        if dm.is_cuda() and _cuda_available():
            backend_type = "cuda"
        elif _cpu_available():
            backend_type = "cpu"
        elif _cuda_available():
            # Fallback to CUDA even if not primary device
            backend_type = "cuda"
            warnings.warn(
                "Using CUDA backend despite non-CUDA device setting. "
                "CPU backend not available.",
                stacklevel=2,
            )
        else:
            raise RuntimeError(
                "No compute backend available. Install CUDA extension or CPU backend."
            )

    # Create backend instance
    if backend_type == "cuda":
        from mast3r_slam.backends.cuda import CUDABackend

        _backend = CUDABackend()
    elif backend_type == "cpu":
        from mast3r_slam.backends.cpu import CPUBackend

        _backend = CPUBackend()
    elif backend_type == "metal":
        raise NotImplementedError("Metal backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    print(f"[Backend] Using {_backend.name} compute backend")
    return _backend


def reset_backend() -> None:
    """Reset cached backend (for testing)."""
    global _backend
    _backend = None


# Re-export base class
__all__ = ["BackendBase", "get_backend", "reset_backend"]
