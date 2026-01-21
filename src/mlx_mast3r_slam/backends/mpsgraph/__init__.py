# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""MPSGraph backend for critical SLAM kernels on Apple Silicon.

This module provides GPU-accelerated implementations of:
- iter_proj: Iterative projection matching with Levenberg-Marquardt
- gauss_newton_rays: Ray-based Gauss-Newton optimization
- refine_matches: Local descriptor-based match refinement

Uses Metal Performance Shaders Graph (MPSGraph) via PyObjC.
"""

from .kernels import (
    iter_proj,
    gauss_newton_rays,
    refine_matches,
    is_available,
)
from .linalg import (
    cholesky_solve,
    solve_linear_system,
    cholesky_decompose,
    solve_2x2,
    solve_3x3,
    sparse_schur_solve,
)

__all__ = [
    "iter_proj",
    "gauss_newton_rays",
    "refine_matches",
    "is_available",
    "cholesky_solve",
    "solve_linear_system",
    "cholesky_decompose",
    "solve_2x2",
    "solve_3x3",
    "sparse_schur_solve",
]
