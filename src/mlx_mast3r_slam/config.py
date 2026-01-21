# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Configuration management for MLX-MASt3R-SLAM."""

from pathlib import Path
from typing import Any

import yaml


config: dict[str, Any] = {}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file with inheritance support."""
    global config
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Handle config inheritance (supports both "inherit" and "_base_" keys)
    inherit_key = None
    if "inherit" in cfg:
        inherit_key = "inherit"
    elif "_base_" in cfg:
        inherit_key = "_base_"

    if inherit_key:
        base_path = Path(cfg[inherit_key])
        if not base_path.is_absolute():
            # Resolve relative to project root, not config file
            base_path = config_path.parent.parent / cfg[inherit_key]
            if not base_path.exists():
                base_path = config_path.parent / cfg[inherit_key]
        base_cfg = load_config(base_path)
        # Merge: base config updated with current config
        _deep_update(base_cfg, cfg)
        cfg = base_cfg
        del cfg[inherit_key]

    config = cfg
    return config


def _deep_update(base: dict, update: dict) -> None:
    """Recursively update base dict with update dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


# Default configuration
DEFAULT_CONFIG: dict[str, Any] = {
    "use_calib": False,
    "single_thread": False,
    "dataset": {
        "img_size": 512,
        "img_downsample": 1,
        "subsample": 1,
        "reverse": False,
    },
    "matching": {
        "use_simple": True,  # Use simple matching (faster, less accurate)
        "use_metal": True,  # Use Metal GPU for iterative projection
        "max_iter": 10,
        "lambda_init": 1e-8,
        "convergence_thresh": 1e-6,
        "dist_thresh": 0.1,
        "radius": 3,
        "dilation_max": 0,
    },
    "tracking": {
        "min_match_frac": 0.05,
        "C_conf": 0.0,
        "Q_conf": 1.5,
        "rel_error": 1e-3,
        "delta_norm": 1e-3,
        "max_iters": 10,
        "huber": 1.345,
        "sigma_ray": 0.003,
        "sigma_dist": 10.0,
        "sigma_pixel": 1.0,
        "sigma_depth": 10.0,
        "pixel_border": 0,
        "depth_eps": 0.0,
        "match_frac_thresh": 0.333,
        "filtering_mode": "weighted_pointmap",
        "filtering_score": "median",
    },
    "local_opt": {
        "window_size": 1000000,
        "pin": 1,
        "max_iters": 10,
        "C_conf": 0.0,
        "Q_conf": 1.5,
        "sigma_ray": 0.003,
        "sigma_dist": 10.0,
        "sigma_pixel": 1.0,
        "sigma_depth": 10.0,
        "pixel_border": 0,
        "depth_eps": 0.0,
        "delta_norm": 1e-3,
    },
    "retrieval": {
        "k": 3,
        "min_thresh": 0.005,
    },
    "reloc": {
        "min_match_frac": 0.3,
        "strict": True,
    },
}


def get_config() -> dict[str, Any]:
    """Get current configuration, using defaults if not loaded."""
    if not config:
        return DEFAULT_CONFIG.copy()
    return config
