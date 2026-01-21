# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Sim3 Lie group operations for pose optimization.

Sim3 represents similarity transformations: rotation + translation + scale.
Parameterization: [tx, ty, tz, qx, qy, qz, qw, s] (8 params)
Lie algebra: [tau_x, tau_y, tau_z, omega_x, omega_y, omega_z, sigma] (7 params)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton convention).

    Args:
        q1, q2: Quaternions [..., 4] as (qx, qy, qz, qw)

    Returns:
        q1 * q2: [..., 4]
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Invert quaternion (conjugate for unit quaternion).

    Args:
        q: Quaternion [..., 4] as (qx, qy, qz, qw)

    Returns:
        q^-1: [..., 4]
    """
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion.

    Args:
        q: Quaternion [..., 4] as (qx, qy, qz, qw)
        v: Vector [..., 3]

    Returns:
        Rotated vector [..., 3]
    """
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # uv = 2 * (q_xyz x v)
    uvx = 2.0 * (qy * vz - qz * vy)
    uvy = 2.0 * (qz * vx - qx * vz)
    uvz = 2.0 * (qx * vy - qy * vx)

    # result = v + qw * uv + q_xyz x uv
    rx = vx + qw * uvx + (qy * uvz - qz * uvy)
    ry = vy + qw * uvy + (qz * uvx - qx * uvz)
    rz = vz + qw * uvz + (qx * uvy - qy * uvx)

    return np.stack([rx, ry, rz], axis=-1)


def sim3_act(t: np.ndarray, q: np.ndarray, s: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply Sim3 transformation to 3D points.

    Args:
        t: Translation [..., 3]
        q: Quaternion [..., 4]
        s: Scale [...] or [..., 1]
        X: Points [..., 3]

    Returns:
        Transformed points: s * R @ X + t
    """
    s_val = s if s.ndim == t.ndim - 1 else s[..., 0]
    Y = quat_rotate(q, X)
    Y = Y * s_val[..., None]
    Y = Y + t
    return Y


def sim3_relative(
    ti: np.ndarray,
    qi: np.ndarray,
    si: np.ndarray,
    tj: np.ndarray,
    qj: np.ndarray,
    sj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute relative transformation Tij = Ti^-1 * Tj.

    Args:
        ti, qi, si: Source pose
        tj, qj, sj: Target pose

    Returns:
        tij, qij, sij: Relative transformation
    """
    # Scale
    si_inv = 1.0 / si
    sij = si_inv * sj

    # Rotation: qij = qi^-1 * qj
    qi_inv = quat_inv(qi)
    qij = quat_multiply(qi_inv, qj)

    # Translation: tij = si^-1 * qi^-1 * (tj - ti)
    tij = tj - ti
    tij = quat_rotate(qi_inv, tij)
    tij = tij * si_inv[..., None]

    return tij, qij, sij


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO3 exponential map: so3 -> SO3 (quaternion).

    Args:
        phi: Rotation vector [..., 3]

    Returns:
        Quaternion [..., 4]
    """
    theta_sq = np.sum(phi * phi, axis=-1)
    theta = np.sqrt(theta_sq + EPS)

    # Taylor expansion for small angles
    half_theta = 0.5 * theta
    small = theta_sq < EPS

    imag = np.where(small, 0.5 - theta_sq / 48.0, np.sin(half_theta) / theta)
    real = np.where(small, 1.0 - theta_sq / 8.0, np.cos(half_theta))

    q = np.stack(
        [
            imag * phi[..., 0],
            imag * phi[..., 1],
            imag * phi[..., 2],
            real,
        ],
        axis=-1,
    )

    return q


def exp_sim3(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sim3 exponential map: sim3 -> Sim3.

    Args:
        xi: Lie algebra element [..., 7] as (tau, omega, sigma)

    Returns:
        t, q, s: Translation [..., 3], quaternion [..., 4], scale [...]
    """
    tau = xi[..., :3]
    omega = xi[..., 3:6]
    sigma = xi[..., 6]

    # Rotation
    q = exp_so3(omega)

    # Scale
    s = np.exp(sigma)

    # Translation (requires computing W matrix)
    theta_sq = np.sum(omega * omega, axis=-1)
    theta = np.sqrt(theta_sq + EPS)

    # Coefficients for W = C*I + A*[omega]_x + B*[omega]_x^2
    small_theta = theta_sq < EPS
    small_sigma = np.abs(sigma) < EPS

    # Default values (for numerical stability)
    C = np.where(small_sigma, 1.0, (s - 1.0) / sigma)

    # A and B coefficients
    A = np.where(
        small_sigma,
        np.where(small_theta, 0.5, (1.0 - np.cos(theta)) / theta_sq),
        np.where(
            small_theta,
            ((sigma - 1.0) * s + 1.0) / (sigma * sigma),
            (s * np.sin(theta) * sigma + (1.0 - s * np.cos(theta)) * theta)
            / (theta * (theta_sq + sigma * sigma)),
        ),
    )

    B = np.where(
        small_sigma,
        np.where(small_theta, 1.0 / 6.0, (theta - np.sin(theta)) / (theta_sq * theta)),
        np.where(
            small_theta,
            (s * 0.5 * sigma * sigma + s - 1.0 - sigma * s) / (sigma * sigma * sigma),
            (
                C
                - ((s * np.cos(theta) - 1.0) * sigma + s * np.sin(theta) * theta)
                / (theta_sq + sigma * sigma)
            )
            / theta_sq,
        ),
    )

    # t = W @ tau = C*tau + A*(omega x tau) + B*(omega x (omega x tau))
    # Cross product omega x tau
    cross1 = np.cross(omega, tau)
    # omega x (omega x tau)
    cross2 = np.cross(omega, cross1)

    t = C[..., None] * tau + A[..., None] * cross1 + B[..., None] * cross2

    return t, q, s


def retract_sim3(
    xi: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retraction on Sim3 manifold: T_new = exp(xi) * T.

    Args:
        xi: Update in Lie algebra [..., 7]
        t, q, s: Current pose

    Returns:
        t_new, q_new, s_new: Updated pose
    """
    dt, dq, ds = exp_sim3(xi)

    # Compose: T_new = dT * T
    q_new = quat_multiply(dq, q)
    t_new = quat_rotate(dq, t) * ds[..., None] + dt
    s_new = ds * s

    return t_new, q_new, s_new


def adjoint_inv_sim3(
    t: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Apply inverse adjoint of Sim3 to a tangent vector.

    This computes Ad^{-1}_T @ X where T = (t, q, s) and X is a 7-vector.
    Used for propagating Jacobians through relative pose computation.

    Args:
        t, q, s: Sim3 pose
        X: Tangent vector [..., 7] as (a, b, c) with a,b in R^3, c in R

    Returns:
        Y: Transformed tangent vector [..., 7]
    """
    a = X[..., :3]  # Translation part
    b = X[..., 3:6]  # Rotation part
    c = X[..., 6:7]  # Scale part

    s_inv = 1.0 / s[..., None] if s.ndim < t.ndim else 1.0 / s

    # Ra = R @ a
    Ra = quat_rotate(q, a)

    # First component: s_inv * R @ a
    Y_a = s_inv * Ra

    # Second component: s_inv * [t]_x @ Ra + R @ b
    Rb = quat_rotate(q, b)
    t_cross_Ra = np.cross(t, Ra)
    Y_b = Rb + s_inv * t_cross_Ra

    # Third component: s_inv * t^T @ Ra + c
    Y_c = c + s_inv * np.sum(t * Ra, axis=-1, keepdims=True)

    return np.concatenate([Y_a, Y_b, Y_c], axis=-1)


def huber_weight(r: np.ndarray, k: float = 1.345) -> np.ndarray:
    """Compute Huber weight for robust optimization.

    Args:
        r: Residuals
        k: Huber threshold (default 1.345 for 95% efficiency)

    Returns:
        Weights
    """
    r_abs = np.abs(r)
    return np.where(r_abs < k, 1.0, k / r_abs)
