# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Main SLAM pipeline for MLX-MASt3R-SLAM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from mlx_mast3r_slam.config import get_config, load_config
from mlx_mast3r_slam.dataloader import Dataset, load_dataset
from mlx_mast3r_slam.frame import Frame, Keyframes, Mode, SLAMState, create_frame
from mlx_mast3r_slam.liegroups import Sim3
from mlx_mast3r_slam.tracker import FrameTracker
from mlx_mast3r_slam.global_opt import FactorGraph
from mlx_mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
    mast3r_match_asymmetric,
    mast3r_match_symmetric,
    RetrievalDatabase,
)


class SLAM:
    """MLX-MASt3R SLAM system.

    Single-threaded implementation optimized for Apple Silicon.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        model_type: str = "dunemast3r",
        model_variant: str = "base",
        resolution: int = 336,
        precision: str = "fp16",
    ):
        """Initialize SLAM system.

        Args:
            config_path: Path to YAML config file
            model_type: MASt3R model type ("dunemast3r" or "mast3r_full")
            model_variant: Encoder variant ("small" or "base")
            resolution: Input resolution
            precision: Model precision
        """
        # Load configuration
        if config_path:
            load_config(config_path)
        self.config = get_config()

        # Load MASt3R model
        print(f"Loading {model_type} model ({model_variant}, {resolution}px)...")
        self.model = load_mast3r(
            model_type=model_type,
            variant=model_variant,
            resolution=resolution,
            precision=precision,
        )
        self.resolution = resolution

        # Initialize components (will be set in run())
        self.keyframes: Optional[Keyframes] = None
        self.tracker: Optional[FrameTracker] = None
        self.factor_graph: Optional[FactorGraph] = None
        self.state: Optional[SLAMState] = None
        self.retrieval_db: Optional[RetrievalDatabase] = None

        # Results storage
        self.timestamps: list[float] = []
        self.poses: list[Sim3] = []

    def run(
        self,
        dataset: Dataset | str | Path,
        callback: Optional[Callable[[Frame, Keyframes], None]] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Run SLAM on dataset.

        Args:
            dataset: Dataset or path to load
            callback: Optional callback for each frame (for visualization)
            max_frames: Maximum frames to process

        Returns:
            Results dict with trajectory and reconstruction
        """
        # Load dataset if path
        if isinstance(dataset, (str, Path)):
            dataset = load_dataset(dataset)

        # Initialize from first frame
        timestamp, img = dataset[0]
        h, w = img.shape[:2]

        # Initialize keyframes storage
        self.keyframes = Keyframes(h, w)

        # Initialize state
        self.state = SLAMState(mode=Mode.INIT)

        # Initialize tracker and factor graph
        self.tracker = FrameTracker(self.model, self.keyframes)

        K = self.keyframes.get_intrinsics() if self.config.get("use_calib") else None
        self.factor_graph = FactorGraph(self.model, self.keyframes, K)

        # Initialize retrieval database for loop closure
        self.retrieval_db = load_retriever(self.model)

        # Clear results
        self.timestamps = []
        self.poses = []

        # Process frames
        n_frames = len(dataset) if max_frames is None else min(len(dataset), max_frames)
        print(f"Processing {n_frames} frames...")

        for i in range(n_frames):
            timestamp, img = dataset[i]

            # Create frame
            frame = create_frame(i, mx.array(img), img_size=self.resolution)

            # Process based on mode
            if self.state.mode == Mode.INIT:
                self._process_init(frame)
            elif self.state.mode == Mode.TRACKING:
                self._process_tracking(frame)
            elif self.state.mode == Mode.RELOC:
                self._process_reloc(frame)

            # Store results
            self.timestamps.append(timestamp)
            self.poses.append(frame.T_WC)

            # Callback for visualization
            if callback:
                callback(frame, self.keyframes)

            # Run global optimization periodically
            self._run_backend()

            # Progress
            if (i + 1) % 10 == 0:
                print(
                    f"Processed {i + 1}/{n_frames} frames, {len(self.keyframes)} keyframes"
                )

        print(f"Done! {len(self.keyframes)} keyframes, {len(self.poses)} poses")

        return self._get_results()

    def _process_init(self, frame: Frame) -> None:
        """Process frame in INIT mode."""
        # Run mono inference
        X, C, feat, pos = mast3r_inference_mono(self.model, frame)

        # Update frame
        frame.X_canon = X
        frame.C = C
        frame.feat = feat
        frame.pos = pos
        frame.N = 1
        frame.N_updates = 1

        # Add as first keyframe
        self.keyframes.append(frame)

        # Add to retrieval database
        self.retrieval_db.update(frame, add_after_query=True)

        # Queue for global optimization
        self.state.queue_global_optimization(0)

        # Switch to tracking mode
        self.state.mode = Mode.TRACKING
        print("Initialized with first keyframe")

    def _process_tracking(self, frame: Frame) -> None:
        """Process frame in TRACKING mode."""
        # Track against last keyframe
        new_kf, match_info, try_reloc = self.tracker.track(
            frame,
            mast3r_match_fn=mast3r_match_asymmetric,
        )

        if try_reloc:
            # Failed to track, try relocalization
            self.state.mode = Mode.RELOC
            self._process_reloc(frame)
            return

        if new_kf:
            # Add new keyframe
            X, C, feat, pos = mast3r_inference_mono(self.model, frame)
            frame.X_canon = X
            frame.C = C
            frame.feat = feat
            frame.pos = pos

            kf_idx = len(self.keyframes)
            self.keyframes.append(frame)

            # Add to retrieval database
            self.retrieval_db.update(frame, add_after_query=True)

            # Queue for global optimization
            self.state.queue_global_optimization(kf_idx)

    def _process_reloc(self, frame: Frame) -> None:
        """Process frame in RELOC mode (relocalization).

        Uses image retrieval to find similar keyframes and attempts
        to establish correspondences for loop closure.
        """
        # Run mono inference
        X, C, feat, pos = mast3r_inference_mono(self.model, frame)
        frame.X_canon = X
        frame.C = C
        frame.feat = feat
        frame.pos = pos

        # Query retrieval database for similar keyframes
        retrieval_config = self.config.get("retrieval", {})
        k = retrieval_config.get("k", 3)
        min_thresh = retrieval_config.get("min_thresh", 0.005)

        similar_kf_indices = self.retrieval_db.update(
            frame,
            add_after_query=False,  # Don't add yet
            k=k,
            min_thresh=min_thresh,
        )

        # Try to relocalize against similar keyframes
        successful_reloc = False

        if similar_kf_indices:
            # Add frame as potential keyframe
            kf_idx = len(self.keyframes)
            self.keyframes.append(frame)

            # Try to add factors against similar keyframes
            reloc_config = self.config.get("reloc", {})
            min_match_frac = reloc_config.get("min_match_frac", 0.3)

            for ref_kf_idx in similar_kf_indices:
                if self.factor_graph.add_factors(
                    [kf_idx],
                    [ref_kf_idx],
                    min_match_frac=min_match_frac,
                    mast3r_match_fn=mast3r_match_symmetric,
                ):
                    successful_reloc = True
                    print(f"Relocalized! Frame {frame.frame_id} -> KF {ref_kf_idx}")

                    # Copy pose from reference keyframe as initial guess
                    frame.T_WC = self.keyframes[ref_kf_idx].T_WC

                    # Add to retrieval database
                    self.retrieval_db.update(frame, add_after_query=True)

                    # Run optimization
                    if self.config.get("use_calib"):
                        self.factor_graph.solve_GN_calib()
                    else:
                        self.factor_graph.solve_GN_rays()
                    break

            if not successful_reloc:
                # Failed to relocalize, remove the frame
                self.keyframes.pop_last()
                print(f"Relocalization failed for frame {frame.frame_id}")
        else:
            # No similar keyframes found, add as new keyframe anyway
            kf_idx = len(self.keyframes)
            self.keyframes.append(frame)
            self.retrieval_db.update(frame, add_after_query=True)
            self.state.queue_global_optimization(kf_idx)
            print(f"No similar keyframes, added frame {frame.frame_id} as new KF")

        # Return to tracking
        self.state.mode = Mode.TRACKING
        self.tracker.reset_idx_f2k()

    def _run_backend(self) -> None:
        """Run backend optimization if needed."""
        while True:
            idx = self.state.dequeue_global_optimization()
            if idx is None:
                break

            # Add factors to graph
            if idx > 0:
                # Connect to previous keyframes
                ii = list(range(max(0, idx - 3), idx))
                jj = [idx] * len(ii)

                if ii:
                    self.factor_graph.add_factors(
                        ii,
                        jj,
                        min_match_frac=self.config["local_opt"].get(
                            "min_match_frac", 0.1
                        ),
                        mast3r_match_fn=mast3r_match_symmetric,
                    )

            # Solve
            if self.config.get("use_calib"):
                self.factor_graph.solve_GN_calib()
            else:
                self.factor_graph.solve_GN_rays()

    def _get_results(self) -> dict:
        """Get SLAM results."""
        # Convert poses to numpy
        pose_matrices = []
        for pose in self.poses:
            T = np.array(pose.matrix())
            pose_matrices.append(T)

        # Get 3D points from keyframes
        points = []
        colors = []
        for kf in self.keyframes._frames:
            if kf.X_canon is not None:
                # Transform points to world frame
                X_world = kf.T_WC.act(kf.X_canon)
                points.append(np.array(X_world))

                # Get colors from image
                img = np.array(kf.img)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                colors.append(img.reshape(-1, 3))

        return {
            "timestamps": np.array(self.timestamps),
            "poses": np.stack(pose_matrices) if pose_matrices else np.array([]),
            "points": np.concatenate(points) if points else np.array([]),
            "colors": np.concatenate(colors) if colors else np.array([]),
            "keyframe_indices": [kf.frame_id for kf in self.keyframes._frames],
        }

    def save_trajectory(self, path: str | Path, format: str = "tum") -> None:
        """Save trajectory to file.

        Args:
            path: Output path
            format: Output format ("tum" or "kitti")
        """
        path = Path(path)

        if format == "tum":
            # TUM format: timestamp tx ty tz qx qy qz qw
            with open(path, "w") as f:
                for ts, pose in zip(self.timestamps, self.poses):
                    t = np.array(pose.translation)
                    q = np.array(pose.rotation.data)  # [qx, qy, qz, qw]
                    f.write(
                        f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
                    )

        elif format == "kitti":
            # KITTI format: 3x4 transformation matrix per row
            with open(path, "w") as f:
                for pose in self.poses:
                    T = np.array(pose.matrix())[:3, :].flatten()
                    f.write(" ".join(f"{x:.6f}" for x in T) + "\n")

        print(f"Saved trajectory to {path}")

    def save_pointcloud(self, path: str | Path) -> None:
        """Save point cloud to PLY file.

        Args:
            path: Output path
        """
        results = self._get_results()
        points = results["points"]
        colors = results["colors"]

        if len(points) == 0:
            print("No points to save")
            return

        path = Path(path)

        # Write PLY
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p, c in zip(points, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        print(f"Saved {len(points)} points to {path}")
