# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Utilities for Lie group conversions.

Provides backward-compatible interface for Sim3 → SE3 conversion.
"""

import einops
import torch

from mast3r_slam.liegroups import SE3, Sim3


def as_SE3(X: Sim3) -> SE3:
    """
    Convert Sim3 to SE3 by dropping the scale component.

    Args:
        X: Sim3 transformation

    Returns:
        SE3 transformation (same rotation and translation, scale=1)
    """
    if isinstance(X, SE3):
        return X

    # Extract t, q (drop scale)
    t, q, _s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    return SE3(torch.cat([t, q], dim=-1))
