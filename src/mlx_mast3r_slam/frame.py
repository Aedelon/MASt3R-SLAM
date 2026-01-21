# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Frame and keyframe data structures for MLX-MASt3R-SLAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mlx.core as mx

from mlx_mast3r_slam.liegroups import Sim3
from mlx_mast3r_slam.config import get_config
from mlx_mast3r_slam.geometry import cartesian_to_spherical, spherical_to_cartesian


class Mode(Enum):
    """SLAM system mode."""

    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclass
class Frame:
    """Single frame data structure.

    Attributes:
        frame_id: Unique frame identifier
        img: Normalized RGB image [3, H, W]
        img_shape: Downsampled image shape [1, 2]
        img_true_shape: Original image shape [1, 2]
        T_WC: World-to-camera pose (Sim3)
        X_canon: Canonical 3D points [H*W, 3]
        C: Per-pixel confidence [H*W, 1]
        feat: MASt3R features [1, num_patches, feat_dim]
        pos: MASt3R positions [1, num_patches, 2]
        N: Number of pointmap updates
        N_updates: Total number of update attempts
        K: Intrinsic matrix [3, 3] (optional)
    """

    frame_id: int
    img: mx.array
    img_shape: mx.array
    img_true_shape: mx.array
    T_WC: Sim3 = field(default_factory=lambda: Sim3.identity())
    X_canon: Optional[mx.array] = None
    C: Optional[mx.array] = None
    feat: Optional[mx.array] = None
    pos: Optional[mx.array] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[mx.array] = None
    _score: Optional[float] = None

    def get_score(self, C: mx.array) -> float:
        """Compute score from confidence map.

        Args:
            C: Confidence values

        Returns:
            Score (median or mean based on config)
        """
        config = get_config()
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            return float(mx.median(C).item())
        else:  # mean
            return float(mx.mean(C).item())

    def update_pointmap(self, X: mx.array, C: mx.array) -> None:
        """Update canonical pointmap with new observations.

        Args:
            X: New 3D points [H*W, 3]
            C: New confidence values [H*W, 1]
        """
        config = get_config()
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X
            self.C = C
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self._score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X
                self.C = C
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X
            self.C = C
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > (self._score or 0.0):
                self.X_canon = X
                self.C = C
                self.N = 1
                self._score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            # Update points where new confidence is higher
            mask_3d = mx.broadcast_to(new_mask, self.X_canon.shape)
            self.X_canon = mx.where(mask_3d, X, self.X_canon)
            self.C = mx.where(new_mask, C, self.C)
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            # Linear weighted average
            total_C = self.C + C
            self.X_canon = (self.C * self.X_canon + C * X) / total_C
            self.C = total_C
            self.N += 1
        elif filtering_mode == "weighted_spherical":
            # Weighted average in spherical coordinates
            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            total_C = self.C + C
            spherical = (self.C * spherical1 + C * spherical2) / total_C
            self.X_canon = spherical_to_cartesian(spherical)
            self.C = total_C
            self.N += 1

        self.N_updates += 1

    def get_average_conf(self) -> Optional[mx.array]:
        """Get average confidence (normalized by N).

        Returns:
            Average confidence or None if no pointmap
        """
        if self.C is None:
            return None
        return self.C / self.N


class Keyframes:
    """Collection of keyframes with shared memory access patterns.

    For MLX, we use simple list-based storage since MLX handles memory efficiently.
    """

    def __init__(
        self,
        h: int,
        w: int,
        buffer_size: int = 512,
        feat_dim: int = 1024,
        patch_size: int = 16,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        """Initialize keyframe storage.

        Args:
            h: Image height
            w: Image width
            buffer_size: Maximum number of keyframes
            feat_dim: Feature dimension
            patch_size: Patch size for feature extraction
            dtype: Data type
        """
        self.h = h
        self.w = w
        self.buffer_size = buffer_size
        self.feat_dim = feat_dim
        self.num_patches = (h * w) // (patch_size * patch_size)
        self.dtype = dtype

        self._frames: list[Frame] = []
        self.K: Optional[mx.array] = None

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> Frame:
        return self._frames[idx]

    def __setitem__(self, idx: int, frame: Frame) -> None:
        if idx >= len(self._frames):
            # Extend list if needed
            self._frames.extend([None] * (idx - len(self._frames) + 1))  # type: ignore
        self._frames[idx] = frame

    def append(self, frame: Frame) -> None:
        """Append a new keyframe."""
        self._frames.append(frame)

    def pop_last(self) -> Optional[Frame]:
        """Remove and return the last keyframe."""
        if self._frames:
            return self._frames.pop()
        return None

    def last_keyframe(self) -> Optional[Frame]:
        """Get the last keyframe without removing it."""
        if self._frames:
            return self._frames[-1]
        return None

    def update_T_WCs(self, T_WCs: Sim3, indices: mx.array) -> None:
        """Update poses for multiple keyframes.

        Args:
            T_WCs: New poses
            indices: Keyframe indices to update
        """
        indices_list = indices.tolist()
        for i, idx in enumerate(indices_list):
            if 0 <= idx < len(self._frames):
                # Extract single pose from batch
                pose_data = T_WCs.data[i : i + 1]
                self._frames[idx].T_WC = Sim3(pose_data)

    def get_poses(self) -> Sim3:
        """Get all keyframe poses as batched Sim3.

        Returns:
            Batched Sim3 poses
        """
        if not self._frames:
            return Sim3.identity()
        pose_data = mx.stack([f.T_WC.data for f in self._frames])
        return Sim3(pose_data.squeeze(1))

    def get_points(self) -> mx.array:
        """Get all keyframe pointmaps stacked.

        Returns:
            Stacked pointmaps [N, H*W, 3]
        """
        if not self._frames:
            return mx.zeros((0, self.h * self.w, 3), dtype=self.dtype)
        return mx.stack([f.X_canon for f in self._frames if f.X_canon is not None])

    def get_confidences(self) -> mx.array:
        """Get all keyframe confidences stacked.

        Returns:
            Stacked confidences [N, H*W, 1]
        """
        if not self._frames:
            return mx.zeros((0, self.h * self.w, 1), dtype=self.dtype)
        return mx.stack([f.get_average_conf() for f in self._frames if f.C is not None])

    def set_intrinsics(self, K: mx.array) -> None:
        """Set shared intrinsic matrix."""
        self.K = K

    def get_intrinsics(self) -> Optional[mx.array]:
        """Get shared intrinsic matrix."""
        return self.K


@dataclass
class SLAMState:
    """Global SLAM state for single-threaded execution.

    For MLX on Apple Silicon, we use single-threaded execution
    since Metal handles parallelism internally.
    """

    mode: Mode = Mode.INIT
    paused: bool = False
    current_frame: Optional[Frame] = None
    global_optimizer_tasks: list[int] = field(default_factory=list)
    reloc_pending: int = 0

    def queue_global_optimization(self, idx: int) -> None:
        """Queue a keyframe for global optimization."""
        self.global_optimizer_tasks.append(idx)

    def dequeue_global_optimization(self) -> Optional[int]:
        """Dequeue a keyframe for global optimization."""
        if self.global_optimizer_tasks:
            return self.global_optimizer_tasks.pop(0)
        return None

    def queue_reloc(self) -> None:
        """Queue relocalization request."""
        self.reloc_pending += 1

    def dequeue_reloc(self) -> bool:
        """Dequeue relocalization request."""
        if self.reloc_pending > 0:
            self.reloc_pending -= 1
            return True
        return False


def create_frame(
    frame_id: int,
    img: mx.array,
    T_WC: Optional[Sim3] = None,
    img_size: int = 512,
) -> Frame:
    """Create a Frame from raw image data.

    Args:
        frame_id: Frame identifier
        img: Raw image [H, W, 3] uint8 or [3, H, W] float
        T_WC: Initial pose (default identity)
        img_size: Target image size

    Returns:
        Initialized Frame
    """
    if T_WC is None:
        T_WC = Sim3.identity()

    # Handle different input formats
    if img.ndim == 3 and img.shape[-1] == 3:
        # [H, W, 3] -> [3, H, W]
        img = mx.transpose(img, (2, 0, 1))

    # Normalize if needed
    if img.dtype == mx.uint8:
        img = img.astype(mx.float32) / 255.0

    h, w = img.shape[1], img.shape[2]
    img_shape = mx.array([[h, w]])
    img_true_shape = mx.array([[h, w]])

    config = get_config()
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        img_shape = img_shape // downsample

    return Frame(
        frame_id=frame_id,
        img=img,
        img_shape=img_shape,
        img_true_shape=img_true_shape,
        T_WC=T_WC,
    )
