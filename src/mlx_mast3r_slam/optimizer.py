# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Non-linear optimization utilities for MLX-MASt3R-SLAM."""

from __future__ import annotations

import math

import mlx.core as mx


def check_convergence(
    iteration: int,
    rel_error_threshold: float,
    delta_norm_threshold: float,
    old_cost: float,
    new_cost: float,
    delta: mx.array,
    verbose: bool = False,
) -> bool:
    """Check if optimization has converged.

    Args:
        iteration: Current iteration number
        rel_error_threshold: Relative error threshold
        delta_norm_threshold: Delta norm threshold
        old_cost: Previous cost
        new_cost: Current cost
        delta: Update vector
        verbose: Print debug info

    Returns:
        True if converged
    """
    cost_diff = old_cost - new_cost
    rel_dec = abs(cost_diff / (old_cost + 1e-10))
    delta_norm = float(mx.sqrt(mx.sum(delta * delta)).item())

    converged = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold

    if verbose:
        print(
            f"{iteration=} | {new_cost=:.6f} {cost_diff=:.6f} "
            f"{rel_dec=:.6f} {delta_norm=:.6f} | {converged=}"
        )

    return converged


def huber_weight(r: mx.array, k: float = 1.345) -> mx.array:
    """Compute Huber robust weights.

    Args:
        r: Residuals
        k: Huber threshold

    Returns:
        Weights for robust estimation
    """
    r_abs = mx.abs(r)
    mask = r_abs < k
    w = mx.where(mask, mx.ones_like(r), k / r_abs)
    return w


def tukey_weight(r: mx.array, t: float = 4.6851) -> mx.array:
    """Compute Tukey robust weights.

    Args:
        r: Residuals
        t: Tukey threshold

    Returns:
        Weights for robust estimation
    """
    r_abs = mx.abs(r)
    tmp = 1 - (r_abs / t) ** 2
    tmp2 = tmp * tmp
    w = mx.where(r_abs < t, tmp2, mx.zeros_like(r))
    return w


def cholesky_solve(H: mx.array, g: mx.array) -> mx.array:
    """Solve H @ x = g using Cholesky decomposition.

    Args:
        H: Symmetric positive definite matrix [..., n, n]
        g: Right-hand side [..., n] or [..., n, 1]

    Returns:
        Solution x [..., n]
    """
    # Add small regularization for numerical stability
    n = H.shape[-1]
    reg = 1e-6 * mx.eye(n, dtype=H.dtype)
    H_reg = H + reg

    # Cholesky decomposition: H = L @ L.T
    L = mx.linalg.cholesky(H_reg)

    # Solve L @ y = g
    # Then L.T @ x = y

    # Forward substitution for L @ y = g
    if g.ndim == H.ndim - 1:
        g = g[..., None]

    # Use triangular solve
    y = mx.linalg.solve_triangular(L, g, upper=False)

    # Backward substitution for L.T @ x = y
    x = mx.linalg.solve_triangular(mx.swapaxes(L, -2, -1), y, upper=True)

    return x.squeeze(-1)


def gauss_newton_step(
    r: mx.array,
    J: mx.array,
    sqrt_info: mx.array,
    huber_k: float = 1.345,
) -> tuple[mx.array, float]:
    """Perform one Gauss-Newton step with robust weighting.

    Args:
        r: Residuals [N, m]
        J: Jacobian [N, m, d] where d is parameter dimension
        sqrt_info: Square root of information matrix [N, m]
        huber_k: Huber threshold

    Returns:
        delta: Parameter update [d]
        cost: Current cost
    """
    # Apply sqrt info
    whitened_r = sqrt_info * r

    # Compute robust weights
    robust_sqrt_info = sqrt_info * mx.sqrt(huber_weight(whitened_r, k=huber_k))

    # Weighted residual and Jacobian
    # A = robust_sqrt_info * J
    # b = robust_sqrt_info * r
    mdim = J.shape[-1]

    # Reshape for matrix multiply
    A = (robust_sqrt_info[..., None] * J).reshape(-1, mdim)
    b = (robust_sqrt_info * r).reshape(-1, 1)

    # Normal equations: H = A.T @ A, g = -A.T @ b
    H = A.T @ A
    g = -A.T @ b

    # Cost
    cost = 0.5 * float((b.T @ b).item())

    # Solve
    delta = cholesky_solve(H, g.squeeze(-1))

    return delta, cost


def batched_cholesky_solve_2x2(H: mx.array, g: mx.array) -> mx.array:
    """Efficient 2x2 Cholesky solve for batched systems.

    Args:
        H: Batch of 2x2 matrices [..., 2, 2]
        g: Batch of 2-vectors [..., 2]

    Returns:
        Solution x [..., 2]
    """
    # For 2x2: direct analytical solution
    a = H[..., 0, 0]
    b = H[..., 0, 1]
    c = H[..., 1, 0]
    d = H[..., 1, 1]

    det = a * d - b * c
    det = mx.maximum(det, 1e-10)

    # Inverse of 2x2 matrix
    inv_det = 1.0 / det
    x0 = inv_det * (d * g[..., 0] - b * g[..., 1])
    x1 = inv_det * (-c * g[..., 0] + a * g[..., 1])

    return mx.stack([x0, x1], axis=-1)


def batched_cholesky_solve(H: mx.array, g: mx.array) -> mx.array:
    """Batched Cholesky solve for small systems.

    Args:
        H: Batch of matrices [..., n, n]
        g: Batch of vectors [..., n]

    Returns:
        Solution x [..., n]
    """
    n = H.shape[-1]

    if n == 2:
        return batched_cholesky_solve_2x2(H, g)

    # General case: use block processing
    batch_shape = H.shape[:-2]
    H_flat = H.reshape(-1, n, n)
    g_flat = g.reshape(-1, n)

    num_systems = H_flat.shape[0]
    x_flat = mx.zeros_like(g_flat)

    # Process each system (could be vectorized further)
    for i in range(num_systems):
        Hi = H_flat[i] + 1e-6 * mx.eye(n, dtype=H.dtype)
        gi = g_flat[i]
        try:
            L = mx.linalg.cholesky(Hi)
            y = mx.linalg.solve_triangular(L, gi[:, None], upper=False)
            xi = mx.linalg.solve_triangular(L.T, y, upper=True)
            x_flat = x_flat.at[i].add(xi.squeeze(-1))
        except Exception:
            # Fallback to pseudo-inverse
            xi = mx.linalg.pinv(Hi) @ gi
            x_flat = x_flat.at[i].add(xi)

    return x_flat.reshape(batch_shape + (n,))


class GaussNewtonOptimizer:
    """Gauss-Newton optimizer for pose estimation."""

    def __init__(
        self,
        max_iters: int = 10,
        rel_error: float = 1e-3,
        delta_norm: float = 1e-3,
        huber_k: float = 1.345,
        verbose: bool = False,
    ) -> None:
        """Initialize optimizer.

        Args:
            max_iters: Maximum iterations
            rel_error: Relative error convergence threshold
            delta_norm: Delta norm convergence threshold
            huber_k: Huber threshold for robust estimation
            verbose: Print debug info
        """
        self.max_iters = max_iters
        self.rel_error = rel_error
        self.delta_norm = delta_norm
        self.huber_k = huber_k
        self.verbose = verbose

    def solve(
        self,
        residual_fn,
        initial_params: mx.array,
        sqrt_info: mx.array,
    ) -> tuple[mx.array, float, int]:
        """Run Gauss-Newton optimization.

        Args:
            residual_fn: Function (params) -> (residuals, jacobian)
            initial_params: Initial parameter values
            sqrt_info: Square root information weights

        Returns:
            params: Optimized parameters
            cost: Final cost
            iterations: Number of iterations used
        """
        params = initial_params
        old_cost = float("inf")

        for i in range(self.max_iters):
            r, J = residual_fn(params)
            delta, new_cost = gauss_newton_step(r, J, sqrt_info, self.huber_k)

            # Update parameters
            params = params + delta

            # Check convergence
            if check_convergence(
                i,
                self.rel_error,
                self.delta_norm,
                old_cost,
                new_cost,
                delta,
                self.verbose,
            ):
                return params, new_cost, i + 1

            old_cost = new_cost

        return params, old_cost, self.max_iters
