# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Device management for multi-backend support (CPU/CUDA/MPS).

This module provides a singleton DeviceManager that handles:
- Auto-detection of the best available device
- CLI argument override for device selection
- Device-specific capabilities and configuration
- Multiprocessing compatibility (IPC device selection)
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from typing import Iterator


class DeviceType(Enum):
    """Supported compute backends."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class DeviceCapabilities:
    """Device-specific capabilities and settings."""

    supports_tf32: bool = False
    supports_amp: bool = False
    supports_share_memory: bool = False
    amp_dtype: torch.dtype = torch.float32
    autocast_device_type: str = "cpu"
    max_memory_gb: Optional[float] = None


class DeviceManager:
    """
    Singleton managing device selection and capabilities.

    Usage:
        from mast3r_slam.device import get_device_manager, get_device

        # Initialize once at startup
        dm = get_device_manager()
        device = dm.initialize(args.device)  # e.g., "cuda:0", "mps", "cpu", or None for auto

        # Get device anywhere in code
        device = get_device()

        # Check capabilities
        if dm.is_cuda():
            # CUDA-specific code
            ...

        # Use autocast
        with dm.autocast_context():
            output = model(input)
    """

    _instance: Optional[DeviceManager] = None

    def __new__(cls) -> DeviceManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._device: Optional[torch.device] = None
        self._device_type: Optional[DeviceType] = None
        self._capabilities: Optional[DeviceCapabilities] = None
        self._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def initialize(self, device_str: Optional[str] = None) -> torch.device:
        """
        Initialize device with auto-detection or CLI override.

        Args:
            device_str: Device override (e.g., "cuda:0", "cuda:1", "mps", "cpu").
                        If None, auto-detects best available device.

        Returns:
            Configured torch.device
        """
        if device_str:
            self._device = torch.device(device_str)
            self._device_type = self._parse_device_type(device_str)
        else:
            self._device, self._device_type = self._auto_detect()

        self._capabilities = self._detect_capabilities()
        self._configure_backends()

        self._log_device_info()

        return self._device

    def _auto_detect(self) -> tuple[torch.device, DeviceType]:
        """Auto-detect best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda:0"), DeviceType.CUDA

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), DeviceType.MPS

        return torch.device("cpu"), DeviceType.CPU

    def _parse_device_type(self, device_str: str) -> DeviceType:
        """Parse device type from string."""
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested ({device_str}) but not available")
            return DeviceType.CUDA
        if device_str == "mps":
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise RuntimeError("MPS requested but not available")
            return DeviceType.MPS
        if device_str == "cpu":
            return DeviceType.CPU
        raise ValueError(f"Unknown device: {device_str}")

    def _detect_capabilities(self) -> DeviceCapabilities:
        """Detect device capabilities."""
        caps = DeviceCapabilities()

        if self._device_type == DeviceType.CUDA:
            caps.supports_tf32 = True
            caps.supports_amp = True
            caps.supports_share_memory = True
            caps.amp_dtype = torch.float16
            caps.autocast_device_type = "cuda"
            if torch.cuda.is_available() and self._device is not None:
                props = torch.cuda.get_device_properties(self._device)
                caps.max_memory_gb = props.total_memory / (1024**3)

        elif self._device_type == DeviceType.MPS:
            caps.supports_tf32 = False
            caps.supports_amp = True
            caps.supports_share_memory = False  # Critical limitation
            caps.amp_dtype = torch.float16
            caps.autocast_device_type = "mps"

        else:  # CPU
            caps.supports_tf32 = False
            caps.supports_amp = False
            caps.supports_share_memory = True
            caps.autocast_device_type = "cpu"

        return caps

    def _configure_backends(self) -> None:
        """Configure PyTorch backends based on device."""
        if self._capabilities is None:
            return

        if self._capabilities.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _log_device_info(self) -> None:
        """Log device information."""
        if self._device is None or self._device_type is None:
            return

        device_name = str(self._device)
        if self._device_type == DeviceType.CUDA and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self._device)
            device_name = (
                f"{self._device} ({props.name}, {props.total_memory / 1024**3:.1f}GB)"
            )
        elif self._device_type == DeviceType.MPS:
            device_name = f"{self._device} (Apple Silicon)"

        print(f"[DeviceManager] Using device: {device_name}")

        if self._device_type == DeviceType.MPS:
            warnings.warn(
                "MPS detected. Some features may use CPU fallback. "
                "For best performance on Apple Silicon, Metal kernels are recommended.",
                stacklevel=2,
            )
        elif self._device_type == DeviceType.CPU:
            warnings.warn(
                "No GPU detected. Running on CPU will be significantly slower.",
                stacklevel=2,
            )

    @property
    def device(self) -> torch.device:
        """Get current torch.device."""
        if self._device is None:
            # Auto-initialize if not done
            return self.initialize()
        return self._device

    @property
    def device_type(self) -> DeviceType:
        """Get current device type."""
        if self._device_type is None:
            self.initialize()
        assert self._device_type is not None
        return self._device_type

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Get device capabilities."""
        if self._capabilities is None:
            self.initialize()
        assert self._capabilities is not None
        return self._capabilities

    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self.device_type == DeviceType.CUDA

    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self.device_type == DeviceType.MPS

    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.device_type == DeviceType.CPU

    @contextmanager
    def autocast_context(self, enabled: bool = True) -> Iterator[None]:
        """
        Context manager for automatic mixed precision.

        Args:
            enabled: Whether to enable autocast

        Yields:
            Context with autocast enabled/disabled
        """
        caps = self.capabilities
        if not caps.supports_amp or not enabled:
            yield
            return

        with torch.amp.autocast(
            device_type=caps.autocast_device_type,
            dtype=caps.amp_dtype,
            enabled=enabled,
        ):
            yield

    def get_ipc_device(self) -> torch.device:
        """
        Get device for inter-process communication (IPC).

        On MPS, tensors must be on CPU for share_memory_() to work.
        On CUDA/CPU, tensors can stay on their device.

        Returns:
            Device suitable for shared memory tensors
        """
        if self.capabilities.supports_share_memory:
            return self.device
        return torch.device("cpu")

    def to_compute_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to compute device."""
        return tensor.to(self.device)

    def to_ipc_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor to IPC device."""
        return tensor.to(self.get_ipc_device())


# Singleton instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global DeviceManager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> torch.device:
    """
    Get the current compute device.

    Convenience function for quick access to the device.
    """
    return get_device_manager().device
