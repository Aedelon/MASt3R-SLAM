// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * CPU Gauss-Newton kernels with OpenMP parallelization.
 *
 * Uses:
 * - OpenMP for parallel Jacobian/Hessian computation
 * - PyTorch torch::linalg for linear system solve
 * - Pure C++ for Lie group operations (Sim3)
 */

#include "gn.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

namespace cpu_backend {

// ============================================================================
// Constants
// ============================================================================

constexpr float EPS = 1e-6f;
constexpr float HUBER_K = 1.345f;

// ============================================================================
// Utility Functions
// ============================================================================

inline float huber_weight(float r) {
    const float r_abs = std::abs(r);
    return r_abs < HUBER_K ? 1.0f : HUBER_K / r_abs;
}

inline float dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline float squared_norm3(const float* v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

inline void cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// ============================================================================
// Quaternion Operations
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

// Relative Sim3: T_ij = T_i^{-1} * T_j
inline void rel_Sim3(
    const float* ti, const float* qi, float si,
    const float* tj, const float* qj, float sj,
    float* tij, float* qij, float* sij)
{
    float si_inv = 1.0f / si;
    *sij = si_inv * sj;

    float qi_inv[4];
    quat_inv(qi, qi_inv);
    quat_mul(qi_inv, qj, qij);

    float dt[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]};
    act_SO3(qi_inv, dt, tij);
    tij[0] *= si_inv;
    tij[1] *= si_inv;
    tij[2] *= si_inv;
}

// Apply Sim3 adjoint inverse for Jacobian transformation
inline void apply_Sim3_adj_inv(
    const float* t, const float* q, float s,
    const float* X,  // input 7-vector
    float* Y)        // output 7-vector
{
    float s_inv = 1.0f / s;

    // First 3: s_inv * R * a
    float a[3] = {X[0], X[1], X[2]};
    float Ra[3];
    act_SO3(q, a, Ra);
    Y[0] = s_inv * Ra[0];
    Y[1] = s_inv * Ra[1];
    Y[2] = s_inv * Ra[2];

    // Next 3: s_inv * [t]x * Ra + R * b
    float b[3] = {X[3], X[4], X[5]};
    float Rb[3];
    act_SO3(q, b, Rb);
    float t_cross_Ra[3];
    cross3(t, Ra, t_cross_Ra);
    Y[3] = Rb[0] + s_inv * t_cross_Ra[0];
    Y[4] = Rb[1] + s_inv * t_cross_Ra[1];
    Y[5] = Rb[2] + s_inv * t_cross_Ra[2];

    // Last: s_inv * t^T * Ra + c
    Y[6] = X[6] + s_inv * dot3(t, Ra);
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
    *s = scale;

    float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
    float theta = std::sqrt(theta_sq);

    float A, B, C;

    if (std::abs(sigma) < EPS) {
        C = 1.0f;
        if (std::abs(theta) < EPS) {
            A = 0.5f;
            B = 1.0f / 6.0f;
        } else {
            A = (1.0f - std::cos(theta)) / theta_sq;
            B = (theta - std::sin(theta)) / (theta_sq * theta);
        }
    } else {
        C = (scale - 1.0f) / sigma;
        if (std::abs(theta) < EPS) {
            float sigma_sq = sigma * sigma;
            A = ((sigma - 1.0f) * scale + 1.0f) / sigma_sq;
            B = (scale * 0.5f * sigma_sq + scale - 1.0f - sigma * scale) / (sigma_sq * sigma);
        } else {
            float a = scale * std::sin(theta);
            float b = scale * std::cos(theta);
            float c = theta_sq + sigma * sigma;
            A = (a * sigma + (1.0f - b) * theta) / (theta * c);
            B = (C - ((b - 1.0f) * sigma + a * theta) / c) / theta_sq;
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

// Retraction: pose_new = exp(xi) * pose
inline void retrSim3(
    const float* xi,
    const float* t, const float* q, float s,
    float* t_new, float* q_new, float* s_new)
{
    float dt[3], dq[4], ds;
    expSim3(xi, dt, dq, &ds);

    // Compose: dT * T
    quat_mul(dq, q, q_new);

    // Normalize quaternion
    float qnorm = std::sqrt(q_new[0]*q_new[0] + q_new[1]*q_new[1] +
                           q_new[2]*q_new[2] + q_new[3]*q_new[3]);
    q_new[0] /= qnorm;
    q_new[1] /= qnorm;
    q_new[2] /= qnorm;
    q_new[3] /= qnorm;

    // t_new = ds * dR @ t + dt
    float rotated_t[3];
    act_SO3(dq, t, rotated_t);
    t_new[0] = ds * rotated_t[0] + dt[0];
    t_new[1] = ds * rotated_t[1] + dt[1];
    t_new[2] = ds * rotated_t[2] + dt[2];

    *s_new = ds * s;
}

// ============================================================================
// Linear System Solve using PyTorch
// ============================================================================

torch::Tensor solve_linear_system(
    torch::Tensor H,    // [N*7, N*7]
    torch::Tensor b,    // [N*7]
    int num_poses,
    float lm_damping = 1e-4f)
{
    const int pose_dim = 7;
    int system_size = num_poses * pose_dim;

    // Add LM damping
    auto diag = H.diagonal();
    H.diagonal().add_(lm_damping * diag.abs() + 1e-6f);

    // Solve using Cholesky
    torch::Tensor dx;
    try {
        auto L = at::linalg_cholesky(H);
        dx = at::cholesky_solve(b.unsqueeze(1), L).squeeze(1);
        dx = -dx;
    } catch (...) {
        // Fallback to general solve
        dx = -at::linalg_solve(H, b);
    }

    return dx.reshape({num_poses, pose_dim});
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

    const int num_edges = ii.size(0);
    const int num_poses = points.size(0);
    const int num_points = points.size(1);
    const int pose_dim = 7;
    const int num_fix = 1;
    const int num_opt = num_poses - num_fix;

    if (num_opt <= 0) {
        return {};
    }

    const float sigma_ray_inv = 1.0f / sigma_ray;
    const float sigma_dist_inv = 1.0f / sigma_dist;

    // Accessors
    auto poses_acc = poses.accessor<float, 2>();
    auto points_acc = points.accessor<float, 3>();
    auto conf_acc = confidences.accessor<float, 3>();
    auto ii_acc = ii.accessor<int64_t, 1>();
    auto jj_acc = jj.accessor<int64_t, 1>();
    auto idx_acc = idx_ii2jj.accessor<int64_t, 2>();
    auto valid_acc = valid_match.accessor<bool, 3>();
    auto Q_acc = Q.accessor<float, 3>();

    // Get unique indices and create mapping
    auto unique_result = torch::_unique(torch::cat({ii, jj}), true);
    auto unique_idx = std::get<0>(unique_result);
    auto ii_opt = torch::searchsorted(unique_idx, ii) - num_fix;
    auto jj_opt = torch::searchsorted(unique_idx, jj) - num_fix;
    auto ii_opt_acc = ii_opt.accessor<int64_t, 1>();
    auto jj_opt_acc = jj_opt.accessor<int64_t, 1>();

    torch::Tensor dx;

    for (int iter = 0; iter < max_iter; iter++) {
        // Allocate Hessian and gradient
        int system_size = num_opt * pose_dim;
        torch::Tensor H = torch::zeros({system_size, system_size}, poses.options());
        torch::Tensor b = torch::zeros({system_size}, poses.options());

        auto H_acc = H.accessor<float, 2>();
        auto b_acc = b.accessor<float, 1>();

        // Process edges in parallel
        #pragma omp parallel for
        for (int e = 0; e < num_edges; e++) {
            int64_t ix = ii_acc[e];
            int64_t jx = jj_acc[e];
            int64_t i_opt = ii_opt_acc[e];
            int64_t j_opt = jj_opt_acc[e];

            // Load poses
            float ti[3] = {poses_acc[ix][0], poses_acc[ix][1], poses_acc[ix][2]};
            float qi[4] = {poses_acc[ix][3], poses_acc[ix][4], poses_acc[ix][5], poses_acc[ix][6]};
            float si = poses_acc[ix][7];

            float tj[3] = {poses_acc[jx][0], poses_acc[jx][1], poses_acc[jx][2]};
            float qj[4] = {poses_acc[jx][3], poses_acc[jx][4], poses_acc[jx][5], poses_acc[jx][6]};
            float sj = poses_acc[jx][7];

            // Relative pose
            float tij[3], qij[4], sij;
            rel_Sim3(ti, qi, si, tj, qj, sj, tij, qij, &sij);

            // Thread-local Hessian and gradient
            float hij[14*14] = {0};
            float vi[7] = {0}, vj[7] = {0};

            // Process points
            for (int k = 0; k < num_points; k++) {
                bool valid_match_k = valid_acc[e][k][0];
                int64_t ind_Xi = valid_match_k ? idx_acc[e][k] : 0;

                float Xi[3] = {points_acc[ix][ind_Xi][0], points_acc[ix][ind_Xi][1], points_acc[ix][ind_Xi][2]};
                float Xj[3] = {points_acc[jx][k][0], points_acc[jx][k][1], points_acc[jx][k][2]};

                // Normalize measurement
                float norm_i = std::sqrt(squared_norm3(Xi));
                float ri[3] = {Xi[0]/norm_i, Xi[1]/norm_i, Xi[2]/norm_i};

                // Transform point
                float Xj_Ci[3];
                act_Sim3(tij, qij, sij, Xj, Xj_Ci);

                float norm_j = std::sqrt(squared_norm3(Xj_Ci));
                float rj_Ci[3] = {Xj_Ci[0]/norm_j, Xj_Ci[1]/norm_j, Xj_Ci[2]/norm_j};

                // Residuals
                float err[4] = {
                    rj_Ci[0] - ri[0],
                    rj_Ci[1] - ri[1],
                    rj_Ci[2] - ri[2],
                    norm_j - norm_i
                };

                // Weights
                float q_val = Q_acc[e][k][0];
                float ci = conf_acc[ix][ind_Xi][0];
                float cj = conf_acc[jx][k][0];

                bool valid = valid_match_k && (q_val > Q_thresh) &&
                            (ci > C_thresh) && (cj > C_thresh);

                float conf_weight = q_val;
                float sqrt_w_ray = valid ? sigma_ray_inv * std::sqrt(conf_weight) : 0.0f;
                float sqrt_w_dist = valid ? sigma_dist_inv * std::sqrt(conf_weight) : 0.0f;

                float w[4];
                w[0] = huber_weight(sqrt_w_ray * err[0]) * sqrt_w_ray * sqrt_w_ray;
                w[1] = huber_weight(sqrt_w_ray * err[1]) * sqrt_w_ray * sqrt_w_ray;
                w[2] = huber_weight(sqrt_w_ray * err[2]) * sqrt_w_ray * sqrt_w_ray;
                w[3] = huber_weight(sqrt_w_dist * err[3]) * sqrt_w_dist * sqrt_w_dist;

                float norm_j_inv = 1.0f / norm_j;
                float norm_j_inv3 = norm_j_inv / (norm_j * norm_j);

                // Jacobians for each residual component
                float Jx[14];
                float* Ji = &Jx[0];
                float* Jj = &Jx[7];

                // rx, ry, rz, dist components
                float drx_dP[3] = {
                    norm_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm_j_inv3,
                    -Xj_Ci[0]*Xj_Ci[1]*norm_j_inv3,
                    -Xj_Ci[0]*Xj_Ci[2]*norm_j_inv3
                };
                float dry_dP[3] = {
                    -Xj_Ci[0]*Xj_Ci[1]*norm_j_inv3,
                    norm_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm_j_inv3,
                    -Xj_Ci[1]*Xj_Ci[2]*norm_j_inv3
                };
                float drz_dP[3] = {
                    -Xj_Ci[0]*Xj_Ci[2]*norm_j_inv3,
                    -Xj_Ci[1]*Xj_Ci[2]*norm_j_inv3,
                    norm_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm_j_inv3
                };

                // rx component
                Ji[0] = drx_dP[0]; Ji[1] = drx_dP[1]; Ji[2] = drx_dP[2];
                Ji[3] = 0.0f; Ji[4] = rj_Ci[2]; Ji[5] = -rj_Ci[1]; Ji[6] = 0.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[0] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[0] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[0] * err[0] * Ji[n];
                    vj[n] += w[0] * err[0] * Jj[n];
                }

                // ry component
                Ji[0] = dry_dP[0]; Ji[1] = dry_dP[1]; Ji[2] = dry_dP[2];
                Ji[3] = -rj_Ci[2]; Ji[4] = 0.0f; Ji[5] = rj_Ci[0]; Ji[6] = 0.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[1] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[1] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[1] * err[1] * Ji[n];
                    vj[n] += w[1] * err[1] * Jj[n];
                }

                // rz component
                Ji[0] = drz_dP[0]; Ji[1] = drz_dP[1]; Ji[2] = drz_dP[2];
                Ji[3] = rj_Ci[1]; Ji[4] = -rj_Ci[0]; Ji[5] = 0.0f; Ji[6] = 0.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[2] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[2] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[2] * err[2] * Ji[n];
                    vj[n] += w[2] * err[2] * Jj[n];
                }

                // dist component
                Ji[0] = rj_Ci[0]; Ji[1] = rj_Ci[1]; Ji[2] = rj_Ci[2];
                Ji[3] = 0.0f; Ji[4] = 0.0f; Ji[5] = 0.0f; Ji[6] = norm_j;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[3] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[3] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[3] * err[3] * Ji[n];
                    vj[n] += w[3] * err[3] * Jj[n];
                }
            }

            // Scatter to global H and b (with atomic add)
            #pragma omp critical
            {
                // Add Hii block
                if (i_opt >= 0) {
                    int i_start = i_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[i_start + n][i_start + m] += hij[n*14 + m];
                        }
                        b_acc[i_start + n] += vi[n];
                    }
                }

                // Add Hjj block
                if (j_opt >= 0) {
                    int j_start = j_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[j_start + n][j_start + m] += hij[(n+7)*14 + (m+7)];
                        }
                        b_acc[j_start + n] += vj[n];
                    }
                }

                // Add off-diagonal blocks
                if (i_opt >= 0 && j_opt >= 0) {
                    int i_start = i_opt * pose_dim;
                    int j_start = j_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[i_start + n][j_start + m] += hij[n*14 + (m+7)];
                            H_acc[j_start + n][i_start + m] += hij[(n+7)*14 + m];
                        }
                    }
                }
            }
        }

        // Solve linear system
        dx = solve_linear_system(H, b, num_opt);

        // Apply retraction
        auto dx_acc = dx.accessor<float, 2>();
        for (int k = num_fix; k < num_poses; k++) {
            float t[3] = {poses_acc[k][0], poses_acc[k][1], poses_acc[k][2]};
            float q[4] = {poses_acc[k][3], poses_acc[k][4], poses_acc[k][5], poses_acc[k][6]};
            float s = poses_acc[k][7];

            float xi[7];
            for (int n = 0; n < 7; n++) {
                xi[n] = dx_acc[k - num_fix][n];
            }

            float t_new[3], q_new[4], s_new;
            retrSim3(xi, t, q, s, t_new, q_new, &s_new);

            poses_acc[k][0] = t_new[0];
            poses_acc[k][1] = t_new[1];
            poses_acc[k][2] = t_new[2];
            poses_acc[k][3] = q_new[0];
            poses_acc[k][4] = q_new[1];
            poses_acc[k][5] = q_new[2];
            poses_acc[k][6] = q_new[3];
            poses_acc[k][7] = s_new;
        }

        // Check convergence
        float delta_norm = dx.norm().item<float>();
        if (delta_norm < delta_thresh) {
            break;
        }
    }

    return {dx};
}

// ============================================================================
// Gauss-Newton for Calibrated Projection
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

    const int num_edges = ii.size(0);
    const int num_poses = points.size(0);
    const int num_points = points.size(1);
    const int pose_dim = 7;
    const int num_fix = 1;
    const int num_opt = num_poses - num_fix;

    if (num_opt <= 0) {
        return {};
    }

    const float sigma_pixel_inv = 1.0f / sigma_pixel;
    const float sigma_depth_inv = 1.0f / sigma_depth;

    // Intrinsics
    auto K_acc = K.accessor<float, 2>();
    float fx = K_acc[0][0];
    float fy = K_acc[1][1];
    float cx = K_acc[0][2];
    float cy = K_acc[1][2];

    // Accessors
    auto poses_acc = poses.accessor<float, 2>();
    auto points_acc = points.accessor<float, 3>();
    auto conf_acc = confidences.accessor<float, 3>();
    auto ii_acc = ii.accessor<int64_t, 1>();
    auto jj_acc = jj.accessor<int64_t, 1>();
    auto idx_acc = idx_ii2jj.accessor<int64_t, 2>();
    auto valid_acc = valid_match.accessor<bool, 3>();
    auto Q_acc = Q.accessor<float, 3>();

    // Optimized indices
    auto unique_result = torch::_unique(torch::cat({ii, jj}), true);
    auto unique_idx = std::get<0>(unique_result);
    auto ii_opt = torch::searchsorted(unique_idx, ii) - num_fix;
    auto jj_opt = torch::searchsorted(unique_idx, jj) - num_fix;
    auto ii_opt_acc = ii_opt.accessor<int64_t, 1>();
    auto jj_opt_acc = jj_opt.accessor<int64_t, 1>();

    torch::Tensor dx;

    for (int iter = 0; iter < max_iter; iter++) {
        int system_size = num_opt * pose_dim;
        torch::Tensor H = torch::zeros({system_size, system_size}, poses.options());
        torch::Tensor b = torch::zeros({system_size}, poses.options());

        auto H_acc = H.accessor<float, 2>();
        auto b_acc = b.accessor<float, 1>();

        #pragma omp parallel for
        for (int e = 0; e < num_edges; e++) {
            int64_t ix = ii_acc[e];
            int64_t jx = jj_acc[e];
            int64_t i_opt = ii_opt_acc[e];
            int64_t j_opt = jj_opt_acc[e];

            // Load poses
            float ti[3] = {poses_acc[ix][0], poses_acc[ix][1], poses_acc[ix][2]};
            float qi[4] = {poses_acc[ix][3], poses_acc[ix][4], poses_acc[ix][5], poses_acc[ix][6]};
            float si = poses_acc[ix][7];

            float tj[3] = {poses_acc[jx][0], poses_acc[jx][1], poses_acc[jx][2]};
            float qj[4] = {poses_acc[jx][3], poses_acc[jx][4], poses_acc[jx][5], poses_acc[jx][6]};
            float sj = poses_acc[jx][7];

            float tij[3], qij[4], sij;
            rel_Sim3(ti, qi, si, tj, qj, sj, tij, qij, &sij);

            float hij[14*14] = {0};
            float vi[7] = {0}, vj[7] = {0};

            for (int k = 0; k < num_points; k++) {
                bool valid_match_k = valid_acc[e][k][0];
                int64_t ind_Xi = valid_match_k ? idx_acc[e][k] : 0;

                float Xi[3] = {points_acc[ix][ind_Xi][0], points_acc[ix][ind_Xi][1], points_acc[ix][ind_Xi][2]};
                float Xj[3] = {points_acc[jx][k][0], points_acc[jx][k][1], points_acc[jx][k][2]};

                // Target pixel
                int u_target = ind_Xi % width;
                int v_target = ind_Xi / width;

                // Transform
                float Xj_Ci[3];
                act_Sim3(tij, qij, sij, Xj, Xj_Ci);

                bool valid_z = (Xj_Ci[2] > z_eps) && (Xi[2] > z_eps);
                float zj_inv = valid_z ? 1.0f / Xj_Ci[2] : 0.0f;

                float x_div_z = Xj_Ci[0] * zj_inv;
                float y_div_z = Xj_Ci[1] * zj_inv;
                float u = fx * x_div_z + cx;
                float v = fy * y_div_z + cy;

                bool valid_u = (u > pixel_border) && (u < width - 1 - pixel_border);
                bool valid_v = (v > pixel_border) && (v < height - 1 - pixel_border);

                float err[3] = {
                    u - (float)u_target,
                    v - (float)v_target,
                    valid_z ? std::log(Xj_Ci[2]) - std::log(Xi[2]) : 0.0f
                };

                float q_val = Q_acc[e][k][0];
                float ci = conf_acc[ix][ind_Xi][0];
                float cj = conf_acc[jx][k][0];

                bool valid = valid_match_k && (q_val > Q_thresh) &&
                            (ci > C_thresh) && (cj > C_thresh) &&
                            valid_u && valid_v && valid_z;

                float conf_weight = q_val;
                float sqrt_w_pixel = valid ? sigma_pixel_inv * std::sqrt(conf_weight) : 0.0f;
                float sqrt_w_depth = valid ? sigma_depth_inv * std::sqrt(conf_weight) : 0.0f;

                float w[3];
                w[0] = huber_weight(sqrt_w_pixel * err[0]) * sqrt_w_pixel * sqrt_w_pixel;
                w[1] = huber_weight(sqrt_w_pixel * err[1]) * sqrt_w_pixel * sqrt_w_pixel;
                w[2] = huber_weight(sqrt_w_depth * err[2]) * sqrt_w_depth * sqrt_w_depth;

                float Jx[14];
                float* Ji = &Jx[0];
                float* Jj = &Jx[7];

                // u residual
                Ji[0] = fx * zj_inv;
                Ji[1] = 0.0f;
                Ji[2] = -fx * x_div_z * zj_inv;
                Ji[3] = -fx * x_div_z * y_div_z;
                Ji[4] = fx * (1.0f + x_div_z * x_div_z);
                Ji[5] = -fx * y_div_z;
                Ji[6] = 0.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[0] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[0] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[0] * err[0] * Ji[n];
                    vj[n] += w[0] * err[0] * Jj[n];
                }

                // v residual
                Ji[0] = 0.0f;
                Ji[1] = fy * zj_inv;
                Ji[2] = -fy * y_div_z * zj_inv;
                Ji[3] = -fy * (1.0f + y_div_z * y_div_z);
                Ji[4] = fy * x_div_z * y_div_z;
                Ji[5] = fy * x_div_z;
                Ji[6] = 0.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[1] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[1] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[1] * err[1] * Ji[n];
                    vj[n] += w[1] * err[1] * Jj[n];
                }

                // depth residual
                Ji[0] = 0.0f;
                Ji[1] = 0.0f;
                Ji[2] = zj_inv;
                Ji[3] = y_div_z;
                Ji[4] = -x_div_z;
                Ji[5] = 0.0f;
                Ji[6] = 1.0f;
                apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
                for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

                for (int n = 0; n < 14; n++) {
                    for (int m = 0; m <= n; m++) {
                        hij[n*14 + m] += w[2] * Jx[n] * Jx[m];
                        if (m != n) hij[m*14 + n] += w[2] * Jx[n] * Jx[m];
                    }
                }
                for (int n = 0; n < 7; n++) {
                    vi[n] += w[2] * err[2] * Ji[n];
                    vj[n] += w[2] * err[2] * Jj[n];
                }
            }

            #pragma omp critical
            {
                if (i_opt >= 0) {
                    int i_start = i_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[i_start + n][i_start + m] += hij[n*14 + m];
                        }
                        b_acc[i_start + n] += vi[n];
                    }
                }

                if (j_opt >= 0) {
                    int j_start = j_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[j_start + n][j_start + m] += hij[(n+7)*14 + (m+7)];
                        }
                        b_acc[j_start + n] += vj[n];
                    }
                }

                if (i_opt >= 0 && j_opt >= 0) {
                    int i_start = i_opt * pose_dim;
                    int j_start = j_opt * pose_dim;
                    for (int n = 0; n < 7; n++) {
                        for (int m = 0; m < 7; m++) {
                            H_acc[i_start + n][j_start + m] += hij[n*14 + (m+7)];
                            H_acc[j_start + n][i_start + m] += hij[(n+7)*14 + m];
                        }
                    }
                }
            }
        }

        dx = solve_linear_system(H, b, num_opt);

        auto dx_acc = dx.accessor<float, 2>();
        for (int k = num_fix; k < num_poses; k++) {
            float t[3] = {poses_acc[k][0], poses_acc[k][1], poses_acc[k][2]};
            float q[4] = {poses_acc[k][3], poses_acc[k][4], poses_acc[k][5], poses_acc[k][6]};
            float s = poses_acc[k][7];

            float xi[7];
            for (int n = 0; n < 7; n++) {
                xi[n] = dx_acc[k - num_fix][n];
            }

            float t_new[3], q_new[4], s_new;
            retrSim3(xi, t, q, s, t_new, q_new, &s_new);

            poses_acc[k][0] = t_new[0];
            poses_acc[k][1] = t_new[1];
            poses_acc[k][2] = t_new[2];
            poses_acc[k][3] = q_new[0];
            poses_acc[k][4] = q_new[1];
            poses_acc[k][5] = q_new[2];
            poses_acc[k][6] = q_new[3];
            poses_acc[k][7] = s_new;
        }

        float delta_norm = dx.norm().item<float>();
        if (delta_norm < delta_thresh) {
            break;
        }
    }

    return {dx};
}

} // namespace cpu_backend
