# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Image processing utilities for MLX-MASt3R-SLAM."""

from __future__ import annotations

import mlx.core as mx


def img_gradient(img: mx.array) -> tuple[mx.array, mx.array]:
    """Compute image gradients using Sobel-like filters.

    Args:
        img: Image tensor [B, C, H, W]

    Returns:
        gx: Gradient in x direction [B, C, H, W]
        gy: Gradient in y direction [B, C, H, W]
    """
    # Simple central difference gradient
    # gx = (img[..., 2:] - img[..., :-2]) / 2
    # gy = (img[..., 2:, :] - img[..., :-2, :]) / 2

    # Pad to maintain size
    b, c, h, w = img.shape

    # X gradient: forward difference with zero padding
    gx = mx.zeros_like(img)
    gx = gx.at[:, :, :, 1:-1].add((img[:, :, :, 2:] - img[:, :, :, :-2]) / 2.0)

    # Y gradient: forward difference with zero padding
    gy = mx.zeros_like(img)
    gy = gy.at[:, :, 1:-1, :].add((img[:, :, 2:, :] - img[:, :, :-2, :]) / 2.0)

    return gx, gy


def bilinear_sample(
    img: mx.array,
    coords: mx.array,
) -> mx.array:
    """Bilinear sampling from image at given coordinates.

    Args:
        img: Image tensor [B, C, H, W]
        coords: Sampling coordinates [B, N, 2] with (x, y) in pixel space

    Returns:
        Sampled values [B, N, C]
    """
    b, c, h, w = img.shape
    n = coords.shape[1]

    x = coords[:, :, 0]
    y = coords[:, :, 1]

    # Get integer coordinates
    x0 = mx.floor(x).astype(mx.int32)
    x1 = x0 + 1
    y0 = mx.floor(y).astype(mx.int32)
    y1 = y0 + 1

    # Clip to image bounds
    x0 = mx.clip(x0, 0, w - 1)
    x1 = mx.clip(x1, 0, w - 1)
    y0 = mx.clip(y0, 0, h - 1)
    y1 = mx.clip(y1, 0, h - 1)

    # Get fractional parts
    fx = x - mx.floor(x)
    fy = y - mx.floor(y)

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w01 = (1 - fx) * fy
    w10 = fx * (1 - fy)
    w11 = fx * fy

    # Expand dims for broadcasting
    w00 = w00[:, :, None]
    w01 = w01[:, :, None]
    w10 = w10[:, :, None]
    w11 = w11[:, :, None]

    # Sample from image
    # img is [B, C, H, W], we need to index [b, :, y, x]
    result = mx.zeros((b, n, c), dtype=img.dtype)

    for bi in range(b):
        # Gather values at 4 corners
        v00 = img[bi, :, y0[bi], x0[bi]].T  # [N, C]
        v01 = img[bi, :, y1[bi], x0[bi]].T
        v10 = img[bi, :, y0[bi], x1[bi]].T
        v11 = img[bi, :, y1[bi], x1[bi]].T

        # Weighted sum
        result = result.at[bi].add(
            w00[bi] * v00 + w01[bi] * v01 + w10[bi] * v10 + w11[bi] * v11
        )

    return result


def resize_image(
    img: mx.array,
    target_size: int | tuple[int, int],
    keep_aspect: bool = True,
) -> mx.array:
    """Resize image to target size.

    Args:
        img: Image [H, W, C] or [C, H, W]
        target_size: Target size (single int or (h, w))
        keep_aspect: Whether to keep aspect ratio

    Returns:
        Resized image
    """
    # Determine input format
    if img.ndim == 3:
        if img.shape[-1] in [1, 3, 4]:
            # [H, W, C] format
            h, w, c = img.shape
            hwc_format = True
        else:
            # [C, H, W] format
            c, h, w = img.shape
            hwc_format = False
    else:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")

    # Determine target dimensions
    if isinstance(target_size, int):
        if keep_aspect:
            scale = target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h = new_w = target_size
    else:
        new_h, new_w = target_size

    # Convert to [H, W, C] if needed
    if not hwc_format:
        img = mx.transpose(img, (1, 2, 0))

    # Simple nearest-neighbor resize for now
    # MLX doesn't have built-in resize, so we use indexing
    y_indices = mx.linspace(0, h - 1, new_h).astype(mx.int32)
    x_indices = mx.linspace(0, w - 1, new_w).astype(mx.int32)

    # Create meshgrid and sample
    yy, xx = mx.meshgrid(y_indices, x_indices, indexing="ij")
    resized = img[yy, xx, :]

    # Convert back to original format
    if not hwc_format:
        resized = mx.transpose(resized, (2, 0, 1))

    return resized
