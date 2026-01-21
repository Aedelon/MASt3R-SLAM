# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Dataset loaders for MLX-MASt3R-SLAM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np

from mlx_mast3r_slam.config import get_config


class Dataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[float, np.ndarray]:
        """Get frame by index.

        Returns:
            timestamp: Frame timestamp
            image: RGB image [H, W, 3] uint8
        """
        pass

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]


class FolderDataset(Dataset):
    """Dataset from folder of images."""

    def __init__(
        self,
        path: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.path = Path(path)
        self.extensions = extensions

        # Find all images
        self.images = sorted(
            [f for f in self.path.iterdir() if f.suffix.lower() in extensions]
        )

        if not self.images:
            raise ValueError(f"No images found in {path} with extensions {extensions}")

        config = get_config()
        self.subsample = config["dataset"].get("subsample", 1)
        self.reverse = config["dataset"].get("reverse", False)

        if self.reverse:
            self.images = self.images[::-1]

    def __len__(self) -> int:
        return len(self.images) // self.subsample

    def __getitem__(self, idx: int) -> tuple[float, np.ndarray]:
        from PIL import Image

        actual_idx = idx * self.subsample
        if actual_idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range")

        img_path = self.images[actual_idx]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        timestamp = float(idx)

        return timestamp, img_array


class TUMDataset(Dataset):
    """TUM RGB-D dataset format."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

        # Read association file
        assoc_file = self.path / "rgb.txt"
        if not assoc_file.exists():
            # Try alternative format
            assoc_file = self.path / "associated.txt"

        self.frames = []
        if assoc_file.exists():
            with open(assoc_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        timestamp = float(parts[0])
                        rgb_path = self.path / parts[1]
                        self.frames.append((timestamp, rgb_path))
        else:
            # Fallback to folder mode
            rgb_dir = self.path / "rgb"
            if rgb_dir.exists():
                for img_path in sorted(rgb_dir.glob("*.png")):
                    timestamp = float(img_path.stem)
                    self.frames.append((timestamp, img_path))

        if not self.frames:
            raise ValueError(f"No frames found in TUM dataset at {path}")

        config = get_config()
        self.subsample = config["dataset"].get("subsample", 1)
        self.reverse = config["dataset"].get("reverse", False)

        if self.reverse:
            self.frames = self.frames[::-1]

    def __len__(self) -> int:
        return len(self.frames) // self.subsample

    def __getitem__(self, idx: int) -> tuple[float, np.ndarray]:
        from PIL import Image

        actual_idx = idx * self.subsample
        timestamp, img_path = self.frames[actual_idx]

        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        return timestamp, img_array


class EuRoCDataset(Dataset):
    """EuRoC MAV dataset format."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

        # Find camera directory
        cam_dir = self.path / "mav0" / "cam0" / "data"
        if not cam_dir.exists():
            cam_dir = self.path / "cam0" / "data"

        if not cam_dir.exists():
            raise ValueError(f"Camera directory not found in EuRoC dataset at {path}")

        self.frames = []
        for img_path in sorted(cam_dir.glob("*.png")):
            timestamp = float(img_path.stem) / 1e9  # nanoseconds to seconds
            self.frames.append((timestamp, img_path))

        if not self.frames:
            raise ValueError(f"No frames found in EuRoC dataset at {path}")

        config = get_config()
        self.subsample = config["dataset"].get("subsample", 1)
        self.reverse = config["dataset"].get("reverse", False)

        if self.reverse:
            self.frames = self.frames[::-1]

    def __len__(self) -> int:
        return len(self.frames) // self.subsample

    def __getitem__(self, idx: int) -> tuple[float, np.ndarray]:
        from PIL import Image

        actual_idx = idx * self.subsample
        timestamp, img_path = self.frames[actual_idx]

        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        return timestamp, img_array


class VideoDataset(Dataset):
    """Dataset from video file (MP4, AVI, etc.)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) required for video datasets")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        config = get_config()
        self.subsample = config["dataset"].get("subsample", 1)
        self.reverse = config["dataset"].get("reverse", False)

        self._cached_frames = {}

    def __len__(self) -> int:
        return self.n_frames // self.subsample

    def __getitem__(self, idx: int) -> tuple[float, np.ndarray]:
        import cv2

        actual_idx = idx * self.subsample
        if self.reverse:
            actual_idx = self.n_frames - 1 - actual_idx

        if actual_idx in self._cached_frames:
            return self._cached_frames[actual_idx]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise IndexError(f"Could not read frame {actual_idx}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = actual_idx / self.fps

        return timestamp, frame

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


def load_dataset(path: str | Path, dataset_type: str | None = None) -> Dataset:
    """Load dataset from path.

    Args:
        path: Dataset path
        dataset_type: Type override ("folder", "tum", "euroc", "video")

    Returns:
        Dataset instance
    """
    path = Path(path)

    if dataset_type is None:
        # Auto-detect dataset type
        if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            dataset_type = "video"
        elif (path / "rgb.txt").exists() or (path / "rgb").exists():
            dataset_type = "tum"
        elif (path / "mav0").exists() or (path / "cam0").exists():
            dataset_type = "euroc"
        else:
            dataset_type = "folder"

    if dataset_type == "folder":
        return FolderDataset(path)
    elif dataset_type == "tum":
        return TUMDataset(path)
    elif dataset_type == "euroc":
        return EuRoCDataset(path)
    elif dataset_type == "video":
        return VideoDataset(path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
