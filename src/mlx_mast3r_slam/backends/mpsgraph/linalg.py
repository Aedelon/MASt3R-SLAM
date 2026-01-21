# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Optimized linear algebra operations using Accelerate (LAPACK on Apple Silicon)."""

from __future__ import annotations

import numpy as np

# scipy.linalg uses Accelerate's LAPACK on macOS
try:
    from scipy import linalg as scipy_linalg

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def cholesky_solve(H: np.ndarray, g: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Solve H @ x = g using optimized solver.

    Uses numpy.linalg.solve which leverages Apple's Accelerate framework
    on macOS for LAPACK operations.

    Args:
        H: Symmetric positive-definite matrix [N, N] or batched [B, N, N]
        g: Right-hand side vector [N] or [N, 1], or batched [B, N] or [B, N, 1]
        reg: Regularization to add to diagonal for numerical stability

    Returns:
        x: Solution vector [N] or batched [B, N]
    """
    # Handle batched case
    if H.ndim == 3:
        B, N, _ = H.shape
        x = np.zeros((B, N), dtype=H.dtype)
        for i in range(B):
            x[i] = cholesky_solve(H[i], g[i] if g.ndim == 2 else g[i].squeeze(), reg)
        return x

    # Ensure g is 1D
    g_vec = g.squeeze() if g.ndim > 1 else g

    # Add regularization
    H_reg = H + reg * np.eye(H.shape[0], dtype=H.dtype)

    # numpy.linalg.solve uses Accelerate's LAPACK on macOS
    try:
        return np.linalg.solve(H_reg, g_vec)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        return np.linalg.lstsq(H_reg, g_vec, rcond=None)[0]


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A @ x = b using optimized LAPACK.

    Args:
        A: Matrix [N, N] or batched [B, N, N]
        b: Right-hand side [N] or batched [B, N]

    Returns:
        x: Solution [N] or batched [B, N]
    """
    if A.ndim == 3:
        B, N, _ = A.shape
        x = np.zeros((B, N), dtype=A.dtype)
        for i in range(B):
            x[i] = solve_linear_system(A[i], b[i])
        return x

    if _SCIPY_AVAILABLE:
        try:
            return scipy_linalg.solve(A, b)
        except Exception:
            pass

    return np.linalg.solve(A, b)


def cholesky_decompose(H: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Cholesky decomposition with regularization.

    Args:
        H: Symmetric positive-definite matrix [N, N] or batched [B, N, N]
        reg: Regularization to add to diagonal

    Returns:
        L: Lower triangular Cholesky factor
    """
    if H.ndim == 3:
        B, N, _ = H.shape
        L = np.zeros_like(H)
        for i in range(B):
            L[i] = cholesky_decompose(H[i], reg)
        return L

    H_reg = H + reg * np.eye(H.shape[0], dtype=H.dtype)

    if _SCIPY_AVAILABLE:
        try:
            return scipy_linalg.cholesky(H_reg, lower=True)
        except Exception:
            pass

    return np.linalg.cholesky(H_reg)


def solve_2x2(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Efficient solve for 2x2 systems (vectorized).

    Args:
        H: Matrices [..., 2, 2]
        g: Right-hand sides [..., 2]

    Returns:
        x: Solutions [..., 2]
    """
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]

    det = a * d - b * c
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    inv_det = 1.0 / det

    x0 = (d * g[..., 0] - b * g[..., 1]) * inv_det
    x1 = (-c * g[..., 0] + a * g[..., 1]) * inv_det

    return np.stack([x0, x1], axis=-1)


def solve_3x3(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Efficient solve for 3x3 systems using Cramer's rule (vectorized).

    Args:
        H: Matrices [..., 3, 3]
        g: Right-hand sides [..., 3]

    Returns:
        x: Solutions [..., 3]
    """
    # Compute determinant and minors for 3x3
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 0, 2]
    d = H[..., 1, 0]
    e = H[..., 1, 1]
    f = H[..., 1, 2]
    g0 = H[..., 2, 0]
    h = H[..., 2, 1]
    i = H[..., 2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g0) + c * (d * h - e * g0)
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    inv_det = 1.0 / det

    # Compute adjugate matrix elements
    A = e * i - f * h
    B = -(d * i - f * g0)
    C = d * h - e * g0
    D = -(b * i - c * h)
    E = a * i - c * g0
    F = -(a * h - b * g0)
    G = b * f - c * e
    H_adj = -(a * f - c * d)
    I = a * e - b * d

    # Solve using adjugate
    x0 = (A * g[..., 0] + D * g[..., 1] + G * g[..., 2]) * inv_det
    x1 = (B * g[..., 0] + E * g[..., 1] + H_adj * g[..., 2]) * inv_det
    x2 = (C * g[..., 0] + F * g[..., 1] + I * g[..., 2]) * inv_det

    return np.stack([x0, x1, x2], axis=-1)


def sparse_schur_solve(
    Hpp: np.ndarray,
    Hpl: np.ndarray,
    Hll: np.ndarray,
    gp: np.ndarray,
    gl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve sparse system using Schur complement.

    For system [Hpp Hpl; Hpl^T Hll] [xp; xl] = [gp; gl]

    Uses Schur complement to eliminate xl first, then back-substitute.

    Args:
        Hpp: Pose-pose block [Np, Np]
        Hpl: Pose-landmark block [Np, Nl]
        Hll: Landmark-landmark block [Nl, Nl] (diagonal)
        gp: Pose gradient [Np]
        gl: Landmark gradient [Nl]

    Returns:
        xp: Pose update [Np]
        xl: Landmark update [Nl]
    """
    # If Hll is diagonal, inversion is trivial
    Hll_inv = 1.0 / (np.diag(Hll) + 1e-10)

    # Schur complement: S = Hpp - Hpl @ Hll_inv @ Hpl^T
    HplHll = Hpl * Hll_inv[None, :]  # [Np, Nl]
    S = Hpp - HplHll @ Hpl.T

    # Modified gradient: gp_s = gp - Hpl @ Hll_inv @ gl
    gp_s = gp - HplHll @ gl

    # Solve for xp
    xp = cholesky_solve(S, gp_s)

    # Back-substitute for xl
    xl = Hll_inv * (gl - Hpl.T @ xp)

    return xp, xl
