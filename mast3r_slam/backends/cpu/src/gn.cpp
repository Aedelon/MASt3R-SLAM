// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * CPU Gauss-Newton kernels with OpenMP parallelization.
 *
 * TODO: Full implementation of GN solver with dense H accumulation
 * and torch.linalg.solve for the linear system.
 */

#include "gn.h"
#include <omp.h>
#include <cmath>

namespace cpu_backend {

// ============================================================================
// Quaternion Operations (for Sim3)
// ============================================================================

inline void quat_inv(const float* q, float* out) {
    out[0] = -q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = q[3];
}

inline void quat_mul(const float* qi, const float* qj, float* out) {
    out[0] = qi[3]*qj[0] + qi[0]*qj[3] + qi[1]*qj[2] - qi[2]*qj[1];
    out[1] = qi[3]*qj[1] - qi[0]*qj[2] + qi[1]*qj[3] + qi[2]*qj[0];
    out[2] = qi[3]*qj[2] + qi[0]*qj[1] - qi[1]*qj[0] + qi[2]*qj[3];
    out[3] = qi[3]*qj[3] - qi[0]*qj[0] - qi[1]*qj[1] - qi[2]*qj[2];
}

inline void act_SO3(const float* q, const float* X, float* Y) {
    float uv[3];
    uv[0] = 2.0f * (q[1]*X[2] - q[2]*X[1]);
    uv[1] = 2.0f * (q[2]*X[0] - q[0]*X[2]);
    uv[2] = 2.0f * (q[0]*X[1] - q[1]*X[0]);

    Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
    Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
    Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

inline void act_Sim3(const float* t, const float* q, float s, const float* X, float* Y) {
    act_SO3(q, X, Y);
    Y[0] = s * Y[0] + t[0];
    Y[1] = s * Y[1] + t[1];
    Y[2] = s * Y[2] + t[2];
}

inline float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// ============================================================================
// Sim3 Exponential Map
// ============================================================================

inline void expSO3(const float* phi, float* q) {
    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float imag, real;

    if (theta_sq < 1e-8f) {
        float theta_p4 = theta_sq * theta_sq;
        imag = 0.5f - (1.0f/48.0f)*theta_sq + (1.0f/3840.0f)*theta_p4;
        real = 1.0f - (1.0f/8.0f)*theta_sq + (1.0f/384.0f)*theta_p4;
    } else {
        float theta = std::sqrt(theta_sq);
        imag = std::sin(0.5f * theta) / theta;
        real = std::cos(0.5f * theta);
    }

    q[0] = imag * phi[0];
    q[1] = imag * phi[1];
    q[2] = imag * phi[2];
    q[3] = real;
}

inline void expSim3(const float* xi, float* t, float* q, float* s) {
    float tau[3] = {xi[0], xi[1], xi[2]};
    float phi[3] = {xi[3], xi[4], xi[5]};
    float sigma = xi[6];

    float scale = std::exp(sigma);
    expSO3(phi, q);
    s[0] = scale;

    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float theta = std::sqrt(theta_sq);

    float A, B, C;
    const float one = 1.0f;
    const float half = 0.5f;
    const float EPS = 1e-6f;

    if (std::abs(sigma) < EPS) {
        C = one;
        if (std::abs(theta) < EPS) {
            A = half;
            B = 1.0f/6.0f;
        } else {
            A = (one - std::cos(theta)) / theta_sq;
            B = (theta - std::sin(theta)) / (theta_sq * theta);
        }
    } else {
        C = (scale - one) / sigma;
        if (std::abs(theta) < EPS) {
            float sigma_sq = sigma * sigma;
            A = ((sigma - one) * scale + one) / sigma_sq;
            B = (scale * half * sigma_sq + scale - one - sigma * scale) / (sigma_sq * sigma);
        } else {
            float a = scale * std::sin(theta);
            float b = scale * std::cos(theta);
            float c = theta_sq + sigma * sigma;
            A = (a * sigma + (one - b) * theta) / (theta * c);
            B = (C - ((b - one) * sigma + a * theta) / c) / theta_sq;
        }
    }

    // t = W @ tau
    t[0] = C * tau[0];
    t[1] = C * tau[1];
    t[2] = C * tau[2];

    float cross1[3];
    cross3(phi, tau, cross1);
    t[0] += A * cross1[0];
    t[1] += A * cross1[1];
    t[2] += A * cross1[2];

    float cross2[3];
    cross3(phi, cross1, cross2);
    t[0] += B * cross2[0];
    t[1] += B * cross2[1];
    t[2] += B * cross2[2];
}

// ============================================================================
// Gauss-Newton for Rays
// ============================================================================

std::vector<torch::Tensor> gauss_newton_rays(
    torch::Tensor poses,
    torch::Tensor points,
    torch::Tensor confidences,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor idx_ii2jj,
    torch::Tensor valid_match,
    torch::Tensor Q,
    float sigma_ray,
    float sigma_dist,
    float C_thresh,
    float Q_thresh,
    int max_iter,
    float delta_thresh)
{
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(confidences);
    CHECK_CONTIGUOUS(ii);
    CHECK_CONTIGUOUS(jj);
    CHECK_CONTIGUOUS(idx_ii2jj);
    CHECK_CONTIGUOUS(valid_match);
    CHECK_CONTIGUOUS(Q);

    // TODO: Full GN implementation
    // Current stub just warns and returns without modifying poses
    TORCH_WARN("gauss_newton_rays: Full CPU implementation pending");
    return {};
}

// ============================================================================
// Gauss-Newton for Calibrated
// ============================================================================

std::vector<torch::Tensor> gauss_newton_calib(
    torch::Tensor poses,
    torch::Tensor points,
    torch::Tensor confidences,
    torch::Tensor K,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor idx_ii2jj,
    torch::Tensor valid_match,
    torch::Tensor Q,
    int height,
    int width,
    int pixel_border,
    float z_eps,
    float sigma_pixel,
    float sigma_depth,
    float C_thresh,
    float Q_thresh,
    int max_iter,
    float delta_thresh)
{
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(confidences);
    CHECK_CONTIGUOUS(K);
    CHECK_CONTIGUOUS(ii);
    CHECK_CONTIGUOUS(jj);
    CHECK_CONTIGUOUS(idx_ii2jj);
    CHECK_CONTIGUOUS(valid_match);
    CHECK_CONTIGUOUS(Q);

    // TODO: Full GN implementation
    TORCH_WARN("gauss_newton_calib: Full CPU implementation pending");
    return {};
}

} // namespace cpu_backend
