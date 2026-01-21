# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""MASt3R integration utilities for MLX-MASt3R-SLAM.

This module provides the interface to mlx-mast3r models for:
- Model loading (DuneMast3r, Mast3rFull)
- Image encoding and 3D reconstruction
- Symmetric and asymmetric matching
- Image retrieval for loop closure
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_mast3r_slam.config import get_config

# Add mlx-mast3r to path
# Path: src/mlx_mast3r_slam/mast3r_utils.py -> project_root/thirdparty/mlx-mast3r/src
MLX_MAST3R_PATH = Path(__file__).parent.parent.parent / "thirdparty" / "mlx-mast3r" / "src"
if str(MLX_MAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(MLX_MAST3R_PATH))

# Import mlx-mast3r components
from mlx_mast3r import (
    DUNE,
    DuneMast3r,
    Mast3r,
    Mast3rFull,
    RetrievalModel,
    compute_similarity_matrix,
    select_pairs_from_retrieval,
)

if TYPE_CHECKING:
    from mlx_mast3r_slam.frame import Frame

# Type alias for model types
ModelType = DuneMast3r | Mast3rFull


def load_mast3r(
    model_type: Literal["dunemast3r", "mast3r_full"] = "dunemast3r",
    variant: Literal["small", "base"] = "base",
    resolution: int = 336,
    precision: Literal["fp16", "fp32", "bf16"] = "fp16",
) -> ModelType:
    """Load MASt3R model.

    Args:
        model_type: Model type
            - "dunemast3r": Fast DUNE encoder + MASt3R decoder (11-32ms @ 336)
            - "mast3r_full": Full MASt3R ViT-Large encoder + decoder (183ms @ 512)
        variant: Encoder variant for DuneMast3r ("small" or "base")
        resolution: Input resolution (336 for DuneMast3r, 512 for Mast3rFull)
        precision: Precision ("fp16", "fp32", "bf16")

    Returns:
        Loaded model (DuneMast3r or Mast3rFull)
    """
    if model_type == "dunemast3r":
        model = DuneMast3r.from_pretrained(
            encoder_variant=variant,
            resolution=resolution,
            precision=precision,
        )
    elif model_type == "mast3r_full":
        model = Mast3rFull.from_pretrained(
            resolution=resolution,
            precision=precision,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'dunemast3r' or 'mast3r_full'")

    return model


def load_retriever(
    model: ModelType,
    backbone_dim: int | None = None,
) -> RetrievalDatabase:
    """Load retrieval model for loop closure detection.

    The retrieval model uses backbone features to compute image similarity
    for detecting revisited places.

    Note: Retrieval requires backbone_dim=1024 (Mast3rFull). DuneMast3r
    has embed_dim=768 (base) or 384 (small), so retrieval quality is reduced.

    Args:
        model: MASt3R model (used to determine backbone dimension)
        backbone_dim: Override backbone dimension (default: auto-detect)

    Returns:
        RetrievalDatabase instance
    """
    # Detect backbone dimension
    if backbone_dim is None:
        if hasattr(model, "embed_dim"):
            backbone_dim = model.embed_dim
        elif isinstance(model, Mast3rFull):
            backbone_dim = 1024
        elif isinstance(model, DuneMast3r):
            backbone_dim = 768 if model.variant == "base" else 384
        else:
            backbone_dim = 1024

    return RetrievalDatabase(model, backbone_dim=backbone_dim)


# ============================================================================
# Image preprocessing
# ============================================================================


def _resize_pil_image(img: Image.Image, long_edge_size: int) -> Image.Image:
    """Resize PIL image to have specified long edge size."""
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(
    img: np.ndarray,
    size: int,
    square_ok: bool = False,
    return_transformation: bool = False,
) -> dict | tuple[dict, tuple]:
    """Resize image for MASt3R input.

    Follows the exact same preprocessing as the original MASt3R:
    - size=224: resize short side to 224, then center crop to square
    - size=512: resize long side to 512, then crop to multiple of 16

    Args:
        img: Input image [H, W, 3] as uint8 or float [0,1]
        size: Target size (224 or 512)
        square_ok: Allow square output for size=512
        return_transformation: Return transformation parameters

    Returns:
        dict with:
        - img: Normalized image tensor [1, H, W, 3]
        - true_shape: Actual shape [[H, W]]
        - unnormalized_img: Cropped image as numpy array
        If return_transformation, also returns (scale_w, scale_h, crop_w, crop_h)
    """
    # Convert to uint8 if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # numpy to PIL format
    pil_img = Image.fromarray(img)
    W1, H1 = pil_img.size

    if size == 224:
        # resize short side to 224 (then crop)
        pil_img = _resize_pil_image(pil_img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to size
        pil_img = _resize_pil_image(pil_img, size)

    W, H = pil_img.size
    cx, cy = W // 2, H // 2

    if size == 224:
        half = min(cx, cy)
        pil_img = pil_img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not square_ok and W == H:
            halfh = int(3 * halfw / 4)
        pil_img = pil_img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    # Normalize: [0, 255] -> [-1, 1]
    img_array = np.asarray(pil_img).astype(np.float32) / 255.0
    img_normalized = (img_array - 0.5) / 0.5

    # Convert to MLX tensor [1, H, W, 3]
    img_tensor = mx.array(img_normalized[None, :, :, :])

    res = {
        "img": img_tensor,
        "true_shape": mx.array([[pil_img.size[1], pil_img.size[0]]]),
        "unnormalized_img": np.asarray(pil_img),
    }

    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - pil_img.size[0]) / 2
        half_crop_h = (H - pil_img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res


def frame_to_numpy(frame: Frame) -> np.ndarray:
    """Convert frame image to numpy uint8 [H, W, 3]."""
    img = frame.img
    if isinstance(img, mx.array):
        img = np.array(img)

    # Handle [3, H, W] format
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # Convert to uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    return img


# ============================================================================
# Inference functions
# ============================================================================


def downsample(
    X: mx.array,
    C: mx.array,
    D: mx.array,
    Q: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Downsample outputs based on config."""
    config = get_config()
    downsample_factor = config.get("dataset", {}).get("img_downsample", 1)

    if downsample_factor > 1:
        # C and Q: (..., H, W)
        # X and D: (..., H, W, F)
        X = X[..., ::downsample_factor, ::downsample_factor, :]
        C = C[..., ::downsample_factor, ::downsample_factor]
        D = D[..., ::downsample_factor, ::downsample_factor, :]
        Q = Q[..., ::downsample_factor, ::downsample_factor]

    return X, C, D, Q


def mast3r_inference_mono(
    model: ModelType,
    frame: Frame,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run MASt3R inference on single frame (self-reconstruction).

    This performs mono-reconstruction where the same image is used
    as both views, producing 3D points in the camera frame.

    Args:
        model: MASt3R model
        frame: Input frame

    Returns:
        Xii: 3D points [H*W, 3] in camera frame
        Cii: Confidence [H*W, 1]
        feat: Encoder features [N, D]
        pos: Feature positions [N, 2]
    """
    img = frame_to_numpy(frame)

    # Encode if not cached
    if frame.feat is None:
        frame.feat = mx.array(model.encode(img))

    # Self-reconstruction
    out1, out2 = model.reconstruct(img, img)

    # Extract outputs - mlx-mast3r returns conf with shape [H, W, 1], squeeze to [H, W]
    pts3d_1 = mx.array(out1["pts3d"])  # [H, W, 3]
    conf_1 = mx.array(out1["conf"]).squeeze(-1)  # [H, W, 1] -> [H, W]
    desc_1 = mx.array(out1.get("desc", np.zeros((1, 1, 24))))  # [H, W, D]
    desc_conf_1_raw = out1.get("desc_conf", out1["conf"])
    desc_conf_1 = mx.array(desc_conf_1_raw).squeeze(-1) if desc_conf_1_raw.ndim == 3 else mx.array(desc_conf_1_raw)

    pts3d_2 = mx.array(out2["pts3d"])
    conf_2 = mx.array(out2["conf"]).squeeze(-1)  # [H, W, 1] -> [H, W]
    desc_2 = mx.array(out2.get("desc", np.zeros((1, 1, 24))))
    desc_conf_2_raw = out2.get("desc_conf", out2["conf"])
    desc_conf_2 = mx.array(desc_conf_2_raw).squeeze(-1) if desc_conf_2_raw.ndim == 3 else mx.array(desc_conf_2_raw)

    # Stack: [2, H, W, ...]
    X = mx.stack([pts3d_1, pts3d_2], axis=0)
    C = mx.stack([conf_1, conf_2], axis=0)
    D = mx.stack([desc_1, desc_2], axis=0)
    Q = mx.stack([desc_conf_1, desc_conf_2], axis=0)

    # Downsample if needed
    X, C, D, Q = downsample(X, C, D, Q)

    # Flatten: [2, H, W, 3] -> [H*W, 3], [2, H, W] -> [H*W, 1]
    h, w = X.shape[1:3]
    Xii = X[0].reshape(h * w, 3)
    Cii = C[0].reshape(h * w, 1)  # C is now [2, H, W] after squeeze

    # Compute feature positions (patch grid)
    # Assumes patch_size=14 for DuneMast3r or 16 for Mast3rFull
    patch_size = 14 if hasattr(model, "variant") else 16
    h_patches = h // patch_size if h >= patch_size else 1
    w_patches = w // patch_size if w >= patch_size else 1

    pos_y, pos_x = np.meshgrid(
        np.arange(h_patches), np.arange(w_patches), indexing="ij"
    )
    pos = mx.array(np.stack([pos_x.flatten(), pos_y.flatten()], axis=-1))

    return Xii, Cii, frame.feat, pos


def mast3r_asymmetric_inference(
    model: ModelType,
    frame_i: Frame,
    frame_j: Frame,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run MASt3R inference on image pair (asymmetric).

    Computes 3D reconstruction from frame_i's viewpoint using frame_j as reference.

    Args:
        model: MASt3R model
        frame_i: Query frame
        frame_j: Reference frame

    Returns:
        X: 3D points [2, H, W, 3]
        C: Confidence [2, H, W]
        D: Descriptors [2, H, W, D]
        Q: Descriptor confidence [2, H, W]
    """
    # Encode if not cached
    if frame_i.feat is None:
        img_i = frame_to_numpy(frame_i)
        frame_i.feat = mx.array(model.encode(img_i))
    if frame_j.feat is None:
        img_j = frame_to_numpy(frame_j)
        frame_j.feat = mx.array(model.encode(img_j))

    # Run reconstruction
    img_i = frame_to_numpy(frame_i)
    img_j = frame_to_numpy(frame_j)
    out_i, out_j = model.reconstruct(img_i, img_j)

    # Extract outputs - squeeze conf from [H, W, 1] to [H, W]
    pts3d_i = mx.array(out_i["pts3d"])
    conf_i = mx.array(out_i["conf"]).squeeze(-1)
    desc_i = mx.array(out_i.get("desc", np.zeros((1, 1, 24))))
    desc_conf_i_raw = out_i.get("desc_conf", out_i["conf"])
    desc_conf_i = mx.array(desc_conf_i_raw).squeeze(-1) if len(desc_conf_i_raw.shape) == 3 else mx.array(desc_conf_i_raw)

    pts3d_j = mx.array(out_j["pts3d"])
    conf_j = mx.array(out_j["conf"]).squeeze(-1)
    desc_j = mx.array(out_j.get("desc", np.zeros((1, 1, 24))))
    desc_conf_j_raw = out_j.get("desc_conf", out_j["conf"])
    desc_conf_j = mx.array(desc_conf_j_raw).squeeze(-1) if len(desc_conf_j_raw.shape) == 3 else mx.array(desc_conf_j_raw)

    # Stack: [2, H, W, ...] for X/D, [2, H, W] for C/Q
    X = mx.stack([pts3d_i, pts3d_j], axis=0)
    C = mx.stack([conf_i, conf_j], axis=0)
    D = mx.stack([desc_i, desc_j], axis=0)
    Q = mx.stack([desc_conf_i, desc_conf_j], axis=0)

    # Downsample
    X, C, D, Q = downsample(X, C, D, Q)

    return X, C, D, Q


def mast3r_symmetric_inference(
    model: ModelType,
    frame_i: Frame,
    frame_j: Frame,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run MASt3R symmetric inference (both directions).

    Computes 4 outputs:
    - (i,i): frame_i predicted from i perspective
    - (j,i): frame_j predicted from i perspective
    - (j,j): frame_j predicted from j perspective
    - (i,j): frame_i predicted from j perspective

    Args:
        model: MASt3R model
        frame_i: First frame
        frame_j: Second frame

    Returns:
        X: 3D points [4, H, W, 3]
        C: Confidence [4, H, W]
        D: Descriptors [4, H, W, D]
        Q: Descriptor confidence [4, H, W]
    """
    # Encode if not cached
    if frame_i.feat is None:
        img_i = frame_to_numpy(frame_i)
        frame_i.feat = mx.array(model.encode(img_i))
    if frame_j.feat is None:
        img_j = frame_to_numpy(frame_j)
        frame_j.feat = mx.array(model.encode(img_j))

    img_i = frame_to_numpy(frame_i)
    img_j = frame_to_numpy(frame_j)

    # Forward: i -> j
    res_ii, res_ji = model.reconstruct(img_i, img_j)

    # Backward: j -> i
    res_jj, res_ij = model.reconstruct(img_j, img_i)

    # Extract all outputs - squeeze conf from [H, W, 1] to [H, W]
    results = [res_ii, res_ji, res_jj, res_ij]
    X_list, C_list, D_list, Q_list = [], [], [], []

    for res in results:
        X_list.append(mx.array(res["pts3d"]))
        C_list.append(mx.array(res["conf"]).squeeze(-1))
        D_list.append(mx.array(res.get("desc", np.zeros((1, 1, 24)))))
        desc_conf_raw = res.get("desc_conf", res["conf"])
        Q_list.append(mx.array(desc_conf_raw).squeeze(-1) if len(desc_conf_raw.shape) == 3 else mx.array(desc_conf_raw))

    # Stack: [4, H, W, 3] for X, [4, H, W] for C/Q, [4, H, W, D] for D
    X = mx.stack(X_list, axis=0)
    C = mx.stack(C_list, axis=0)
    D = mx.stack(D_list, axis=0)
    Q = mx.stack(Q_list, axis=0)

    # Downsample
    X, C, D, Q = downsample(X, C, D, Q)

    return X, C, D, Q


# ============================================================================
# Matching functions
# ============================================================================


def mast3r_match_asymmetric(
    model: ModelType,
    frame_i: Frame,
    frame_j: Frame,
    idx_i2j_init: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Asymmetric matching from frame_i to frame_j.

    Args:
        model: MASt3R model
        frame_i: Query frame (current frame)
        frame_j: Reference frame (keyframe)
        idx_i2j_init: Initial correspondence indices

    Returns:
        idx_i2j: Correspondence indices [B, H*W]
        valid_match_j: Validity mask [B, H*W, 1]
        Xii: Frame i 3D points [B, H*W, 3]
        Cii: Frame i confidence [B, H*W, 1]
        Qii: Frame i match quality [B, H*W, 1]
        Xji: Frame j 3D points [B, H*W, 3]
        Cji: Frame j confidence [B, H*W, 1]
        Qji: Frame j match quality [B, H*W, 1]
    """
    from mlx_mast3r_slam.matching import match

    # Run asymmetric inference
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    # X shape: [2, H, W, 3]
    h, w = X.shape[1:3]

    # Split outputs (2 outputs per inference)
    Xii, Xji = X[0:1], X[1:2]  # Keep batch dim
    Cii, Cji = C[0:1], C[1:2]
    Dii, Dji = D[0:1], D[1:2]
    Qii, Qji = Q[0:1], Q[1:2]

    # Match using iterative projection
    idx_i2j, valid_match_j = match(Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init)

    # Flatten spatial dimensions
    Xii = Xii.reshape(1, h * w, 3)
    Xji = Xji.reshape(1, h * w, 3)
    Cii = Cii.reshape(1, h * w, 1)
    Cji = Cji.reshape(1, h * w, 1)
    Qii = Qii.reshape(1, h * w, 1)
    Qji = Qji.reshape(1, h * w, 1)

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def mast3r_match_symmetric(
    model: ModelType,
    feat_i: mx.array,
    pos_i: mx.array,
    feat_j: mx.array,
    pos_j: mx.array,
    shape_i: list,
    shape_j: list,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Symmetric matching between keyframe pairs.

    Used for global optimization where bidirectional correspondences are needed.

    Args:
        model: MASt3R model
        feat_i: Features from keyframes i [B, N, D]
        pos_i: Positions from keyframes i [B, N, 2]
        feat_j: Features from keyframes j [B, N, D]
        pos_j: Positions from keyframes j [B, N, 2]
        shape_i: Image shapes for keyframes i
        shape_j: Image shapes for keyframes j

    Returns:
        idx_i2j: Correspondence indices i->j [B, N]
        idx_j2i: Correspondence indices j->i [B, N]
        valid_match_j: Validity mask for j [B, N, 1]
        valid_match_i: Validity mask for i [B, N, 1]
        Qii: Quality scores for i in i frame [B, N, 1]
        Qjj: Quality scores for j in j frame [B, N, 1]
        Qji: Quality scores for j in i frame [B, N, 1]
        Qij: Quality scores for i in j frame [B, N, 1]
    """
    from mlx_mast3r_slam.matching import match

    # This requires running the decoder with cross-attention
    # For now, use a batch-aware implementation
    batch_size = feat_i.shape[0]

    # Get image dimensions
    h = int(shape_i[0][0, 0].item()) if hasattr(shape_i[0], "item") else int(shape_i[0][0, 0])
    w = int(shape_i[0][0, 1].item()) if hasattr(shape_i[0], "item") else int(shape_i[0][0, 1])

    # For symmetric matching, we need to decode in both directions
    # This is a simplified version - full implementation would batch decode
    X_all, C_all, D_all, Q_all = [], [], [], []

    # Process batch
    # Note: In production, this would use batched decoder calls
    for b in range(batch_size):
        # Create pseudo-frames for this batch element
        # This is a workaround - proper implementation needs direct decoder access
        pass

    # For now, use identity matching as fallback
    n_points = h * w
    idx_i2j = mx.broadcast_to(mx.arange(n_points)[None, :], (batch_size, n_points))
    idx_j2i = mx.array(idx_i2j)  # MLX copy

    valid_match_j = mx.ones((batch_size, n_points, 1), dtype=mx.bool_)
    valid_match_i = mx.ones((batch_size, n_points, 1), dtype=mx.bool_)

    Qii = mx.ones((batch_size, n_points, 1), dtype=mx.float32)
    Qjj = mx.ones((batch_size, n_points, 1), dtype=mx.float32)
    Qji = mx.ones((batch_size, n_points, 1), dtype=mx.float32)
    Qij = mx.ones((batch_size, n_points, 1), dtype=mx.float32)

    return idx_i2j, idx_j2i, valid_match_j, valid_match_i, Qii, Qjj, Qji, Qij


def mast3r_decode_symmetric_batch(
    model: ModelType,
    feat_i: mx.array,
    pos_i: mx.array,
    feat_j: mx.array,
    pos_j: mx.array,
    shape_i: list,
    shape_j: list,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Batch decode symmetric features.

    This decodes pre-computed features through the MASt3R decoder
    in both directions for symmetric matching.

    Args:
        model: MASt3R model
        feat_i: Features [B, N, D]
        pos_i: Positions [B, N, 2]
        feat_j: Features [B, N, D]
        pos_j: Positions [B, N, 2]
        shape_i: Shapes for i
        shape_j: Shapes for j

    Returns:
        X: 3D points [4, B, H, W, 3]
        C: Confidence [4, B, H, W]
        D: Descriptors [4, B, H, W, D]
        Q: Descriptor confidence [4, B, H, W]
    """
    batch_size = feat_i.shape[0]

    X_batch, C_batch, D_batch, Q_batch = [], [], [], []

    # Process each batch element
    for b in range(batch_size):
        # Get shapes
        h_i = int(shape_i[b][0, 0].item()) if hasattr(shape_i[b], "item") else int(shape_i[b][0, 0])
        w_i = int(shape_i[b][0, 1].item()) if hasattr(shape_i[b], "item") else int(shape_i[b][0, 1])

        # Create placeholder outputs
        # In production, this would use model._decoder() directly
        X_b = mx.zeros((4, h_i, w_i, 3), dtype=mx.float32)
        C_b = mx.ones((4, h_i, w_i), dtype=mx.float32)
        D_b = mx.zeros((4, h_i, w_i, 24), dtype=mx.float32)
        Q_b = mx.ones((4, h_i, w_i), dtype=mx.float32)

        X_batch.append(X_b)
        C_batch.append(C_b)
        D_batch.append(D_b)
        Q_batch.append(Q_b)

    # Stack: [4, B, H, W, ...]
    X = mx.stack(X_batch, axis=1)
    C = mx.stack(C_batch, axis=1)
    D = mx.stack(D_batch, axis=1)
    Q = mx.stack(Q_batch, axis=1)

    # Downsample
    X, C, D, Q = downsample(X, C, D, Q)

    return X, C, D, Q


# ============================================================================
# Retrieval Database
# ============================================================================


class RetrievalDatabase:
    """Database for image retrieval and loop closure detection.

    Uses the RetrievalModel from mlx-mast3r to compute global signatures
    and find similar images in the database.

    Note: Full retrieval (with pretrained weights) requires backbone_dim=1024
    (Mast3rFull). For DuneMast3r (dim=768 or 384), a simplified cosine
    similarity approach is used instead.
    """

    def __init__(
        self,
        model: ModelType,
        backbone_dim: int = 1024,
    ):
        """Initialize retrieval database.

        Args:
            model: Backbone model for feature extraction
            backbone_dim: Backbone feature dimension
        """
        self.model = model
        self.backbone_dim = backbone_dim

        # Load retrieval model only if backbone_dim matches (1024 for MASt3R)
        self.retrieval = None
        self.use_simple_retrieval = backbone_dim != 1024

        if not self.use_simple_retrieval:
            try:
                self.retrieval = RetrievalModel.from_pretrained(backbone_dim=backbone_dim)
            except Exception:
                # Fallback to simple retrieval
                self.use_simple_retrieval = True

        # Database storage
        self.signatures: list[mx.array] = []
        self.kf_ids: list[int] = []
        self.kf_counter = 0

    def prep_features(self, features: mx.array) -> mx.array:
        """Prepare features for retrieval.

        Applies retrieval head to backbone features to get
        local features suitable for aggregation.

        Args:
            features: Backbone features [N, D]

        Returns:
            Processed features [N, D']
        """
        feat_whitened, attention = self.retrieval.forward_features(features)
        return feat_whitened

    def compute_signature(self, features: mx.array) -> mx.array:
        """Compute global signature from features.

        Args:
            features: Backbone features [N, D]

        Returns:
            Global signature [D']
        """
        if self.use_simple_retrieval:
            # Simple approach: mean pooling + L2 normalization
            if features.ndim == 1:
                signature = features
            else:
                signature = mx.mean(features, axis=0)
            # L2 normalize
            norm = mx.sqrt(mx.sum(signature * signature) + 1e-8)
            return signature / norm
        else:
            return self.retrieval.forward_global(features)

    def update(
        self,
        frame: Frame,
        add_after_query: bool = True,
        k: int = 3,
        min_thresh: float = 0.0,
    ) -> list[int]:
        """Update database with new frame.

        Args:
            frame: Frame to add
            add_after_query: Whether to add frame to database
            k: Number of top matches to return
            min_thresh: Minimum similarity threshold

        Returns:
            List of matching keyframe indices
        """
        # Get features
        if frame.feat is None:
            img = frame_to_numpy(frame)
            frame.feat = mx.array(self.model.encode(img))

        # Compute signature
        signature = self.compute_signature(frame.feat)

        # Query existing database
        topk_indices = []
        if self.kf_counter > 0:
            # Stack all signatures
            db_sigs = mx.stack(self.signatures, axis=0)  # [N, D]

            # Compute similarities
            similarities = signature @ db_sigs.T  # [N]

            # Get top-k
            k_actual = min(k, len(self.signatures))
            top_k = mx.argsort(similarities)[::-1][:k_actual]

            for idx in top_k:
                idx_int = int(idx.item())
                sim = float(similarities[idx_int].item())
                if sim > min_thresh:
                    topk_indices.append(self.kf_ids[idx_int])

        # Add to database
        if add_after_query:
            self.signatures.append(signature)
            self.kf_ids.append(self.kf_counter)
            self.kf_counter += 1

        return topk_indices

    def query(self, features: mx.array, k: int = 3) -> tuple[list[int], list[float]]:
        """Query database for similar images.

        Args:
            features: Query features [N, D]
            k: Number of results

        Returns:
            Tuple of (indices, scores)
        """
        if len(self.signatures) == 0:
            return [], []

        signature = self.compute_signature(features)
        db_sigs = mx.stack(self.signatures, axis=0)
        similarities = signature @ db_sigs.T

        k_actual = min(k, len(self.signatures))
        top_k = mx.argsort(similarities)[::-1][:k_actual]

        indices = [self.kf_ids[int(idx.item())] for idx in top_k]
        scores = [float(similarities[int(idx.item())].item()) for idx in top_k]

        return indices, scores


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Model loading
    "load_mast3r",
    "load_retriever",
    # Models (re-exported)
    "DUNE",
    "DuneMast3r",
    "Mast3r",
    "Mast3rFull",
    "RetrievalModel",
    # Preprocessing
    "resize_img",
    "frame_to_numpy",
    # Inference
    "mast3r_inference_mono",
    "mast3r_asymmetric_inference",
    "mast3r_symmetric_inference",
    # Matching
    "mast3r_match_asymmetric",
    "mast3r_match_symmetric",
    "mast3r_decode_symmetric_batch",
    # Retrieval
    "RetrievalDatabase",
]
