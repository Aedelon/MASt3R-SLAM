// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Gauss-Newton optimization kernels for Metal.
 *
 * Implements ray alignment and calibrated projection residuals for
 * Sim3 pose optimization in SLAM.
 *
 * Architecture:
 * - Metal kernels compute Jacobians and Hessian blocks (parallel)
 * - PyTorch MPS solves the linear system (replaces Eigen)
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint THREADS = 256;
constant float EPS = 1e-6f;
constant float HUBER_K = 1.345f;

// ============================================================================
// Structures
// ============================================================================

struct GNParams {
    uint num_points;       // Points per edge
    uint num_edges;        // Number of edges
    float sigma_ray;       // Ray error weight
    float sigma_dist;      // Distance error weight
    float C_thresh;        // Confidence threshold
    float Q_thresh;        // Match quality threshold
};

struct CalibParams {
    uint num_points;
    uint num_edges;
    uint height;
    uint width;
    uint pixel_border;
    float z_eps;
    float sigma_pixel;
    float sigma_depth;
    float C_thresh;
    float Q_thresh;
};

// ============================================================================
// Utility Functions
// ============================================================================

inline float huber(float r) {
    const float r_abs = abs(r);
    return r_abs < HUBER_K ? 1.0f : HUBER_K / r_abs;
}

inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float squared_norm3(float3 v) {
    return dot3(v, v);
}

// Quaternion multiplication: qi * qj
// q = [x, y, z, w] format
inline float4 quat_mul(float4 qi, float4 qj) {
    return float4(
        qi.w * qj.x + qi.x * qj.w + qi.y * qj.z - qi.z * qj.y,
        qi.w * qj.y - qi.x * qj.z + qi.y * qj.w + qi.z * qj.x,
        qi.w * qj.z + qi.x * qj.y - qi.y * qj.x + qi.z * qj.w,
        qi.w * qj.w - qi.x * qj.x - qi.y * qj.y - qi.z * qj.z
    );
}

// Quaternion inverse (conjugate for unit quaternion)
inline float4 quat_inv(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

// Rotate vector by quaternion: q * v * q^-1
inline float3 rotate_by_quat(float4 q, float3 v) {
    float3 u = float3(q.x, q.y, q.z);
    float3 uv = 2.0f * cross(u, v);
    return v + q.w * uv + cross(u, uv);
}

// Sim3 action: s * R * X + t
inline float3 act_sim3(float3 t, float4 q, float s, float3 X) {
    float3 rotated = rotate_by_quat(q, X);
    return s * rotated + t;
}

// Relative Sim3: T_ij = T_i^{-1} * T_j
inline void rel_sim3(
    float3 ti, float4 qi, float si,
    float3 tj, float4 qj, float sj,
    thread float3& tij, thread float4& qij, thread float& sij
) {
    float si_inv = 1.0f / si;
    sij = si_inv * sj;

    float4 qi_inv = quat_inv(qi);
    qij = quat_mul(qi_inv, qj);

    float3 dt = tj - ti;
    tij = si_inv * rotate_by_quat(qi_inv, dt);
}

// Apply Sim3 adjoint inverse (for Jacobian transformation)
// X is 7-vector [tau(3), omega(3), sigma(1)]
inline void apply_sim3_adj_inv(
    float3 t, float4 q, float s,
    thread float* X,  // input 7-vector
    thread float* Y   // output 7-vector
) {
    float s_inv = 1.0f / s;

    // First 3 components: s_inv * R * a
    float3 a = float3(X[0], X[1], X[2]);
    float3 Ra = rotate_by_quat(q, a);
    Y[0] = s_inv * Ra.x;
    Y[1] = s_inv * Ra.y;
    Y[2] = s_inv * Ra.z;

    // Next 3 components: s_inv * [t]x * Ra + R * b
    float3 b = float3(X[3], X[4], X[5]);
    float3 Rb = rotate_by_quat(q, b);
    float3 t_cross_Ra = cross(t, Ra);
    Y[3] = Rb.x + s_inv * t_cross_Ra.x;
    Y[4] = Rb.y + s_inv * t_cross_Ra.y;
    Y[5] = Rb.z + s_inv * t_cross_Ra.z;

    // Last component: s_inv * t^T * Ra + c
    Y[6] = X[6] + s_inv * dot3(t, Ra);
}

// ============================================================================
// SO3 Exponential Map
// ============================================================================

inline float4 exp_so3(float3 phi) {
    float theta_sq = squared_norm3(phi);

    float imag, real;
    if (theta_sq < EPS) {
        float theta_p4 = theta_sq * theta_sq;
        imag = 0.5f - (1.0f/48.0f) * theta_sq + (1.0f/3840.0f) * theta_p4;
        real = 1.0f - (1.0f/8.0f) * theta_sq + (1.0f/384.0f) * theta_p4;
    } else {
        float theta = sqrt(theta_sq);
        imag = sin(0.5f * theta) / theta;
        real = cos(0.5f * theta);
    }

    return float4(imag * phi.x, imag * phi.y, imag * phi.z, real);
}

// ============================================================================
// Sim3 Exponential Map
// ============================================================================

inline void exp_sim3(
    thread float* xi,  // 7-vector [tau, omega, sigma]
    thread float3& t,
    thread float4& q,
    thread float& s
) {
    float3 tau = float3(xi[0], xi[1], xi[2]);
    float3 phi = float3(xi[3], xi[4], xi[5]);
    float sigma = xi[6];

    float scale = exp(sigma);
    q = exp_so3(phi);
    s = scale;

    float theta_sq = squared_norm3(phi);
    float theta = sqrt(theta_sq);

    float A, B, C;

    if (abs(sigma) < EPS) {
        C = 1.0f;
        if (abs(theta) < EPS) {
            A = 0.5f;
            B = 1.0f / 6.0f;
        } else {
            A = (1.0f - cos(theta)) / theta_sq;
            B = (theta - sin(theta)) / (theta_sq * theta);
        }
    } else {
        C = (scale - 1.0f) / sigma;
        if (abs(theta) < EPS) {
            float sigma_sq = sigma * sigma;
            A = ((sigma - 1.0f) * scale + 1.0f) / sigma_sq;
            B = (scale * 0.5f * sigma_sq + scale - 1.0f - sigma * scale) / (sigma_sq * sigma);
        } else {
            float a = scale * sin(theta);
            float b = scale * cos(theta);
            float c = theta_sq + sigma * sigma;
            A = (a * sigma + (1.0f - b) * theta) / (theta * c);
            B = (C - ((b - 1.0f) * sigma + a * theta) / c) / theta_sq;
        }
    }

    // W = C * I + A * Phi + B * Phi^2
    // t = W * tau
    t = C * tau;

    float3 phi_cross_tau = cross(phi, tau);
    t += A * phi_cross_tau;

    float3 phi_cross_phi_cross_tau = cross(phi, phi_cross_tau);
    t += B * phi_cross_phi_cross_tau;
}

// ============================================================================
// Sim3 Retraction
// ============================================================================

inline void retr_sim3(
    thread float* xi,
    float3 t, float4 q, float s,
    thread float3& t1, thread float4& q1, thread float& s1
) {
    float3 dt;
    float4 dq;
    float ds;

    exp_sim3(xi, dt, dq, ds);

    // Compose from left: T1 = dT * T
    q1 = quat_mul(dq, q);
    t1 = ds * rotate_by_quat(dq, t) + dt;
    s1 = ds * s;
}

// ============================================================================
// Pose Retraction Kernel
// ============================================================================

kernel void pose_retr_kernel(
    device float* poses [[buffer(0)]],           // [num_poses, 8]
    device const float* dx [[buffer(1)]],        // [num_poses - num_fix, 7]
    constant uint& num_poses [[buffer(2)]],
    constant uint& num_fix [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint k = num_fix + tid;
    if (k >= num_poses) return;

    uint pose_idx = k * 8;
    uint dx_idx = (k - num_fix) * 7;

    // Load current pose
    float3 t = float3(poses[pose_idx], poses[pose_idx + 1], poses[pose_idx + 2]);
    float4 q = float4(poses[pose_idx + 3], poses[pose_idx + 4],
                      poses[pose_idx + 5], poses[pose_idx + 6]);
    float s = poses[pose_idx + 7];

    // Load delta
    float xi[7];
    for (int n = 0; n < 7; n++) {
        xi[n] = dx[dx_idx + n];
    }

    // Apply retraction
    float3 t1;
    float4 q1;
    float s1;
    retr_sim3(xi, t, q, s, t1, q1, s1);

    // Store updated pose
    poses[pose_idx] = t1.x;
    poses[pose_idx + 1] = t1.y;
    poses[pose_idx + 2] = t1.z;
    poses[pose_idx + 3] = q1.x;
    poses[pose_idx + 4] = q1.y;
    poses[pose_idx + 5] = q1.z;
    poses[pose_idx + 6] = q1.w;
    poses[pose_idx + 7] = s1;
}

// ============================================================================
// Ray Alignment Kernel
// Computes Jacobians and Hessian blocks for ray-based alignment
// ============================================================================

kernel void ray_align_kernel(
    device const float* Twc [[buffer(0)]],           // [num_poses, 8]
    device const float* Xs [[buffer(1)]],            // [num_poses, num_points, 3]
    device const float* Cs [[buffer(2)]],            // [num_poses, num_points, 1]
    device const int64_t* ii [[buffer(3)]],          // [num_edges]
    device const int64_t* jj [[buffer(4)]],          // [num_edges]
    device const int64_t* idx_ii2jj [[buffer(5)]],   // [num_edges, num_points]
    device const bool* valid_match [[buffer(6)]],    // [num_edges, num_points, 1]
    device const float* Q [[buffer(7)]],             // [num_edges, num_points, 1]
    device float* Hs [[buffer(8)]],                  // [4, num_edges, 7, 7]
    device float* gs [[buffer(9)]],                  // [2, num_edges, 7]
    constant GNParams& params [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]  // [THREADS]
) {
    const uint num_points = params.num_points;
    const float sigma_ray_inv = 1.0f / params.sigma_ray;
    const float sigma_dist_inv = 1.0f / params.sigma_dist;

    // Get pose indices for this edge
    int64_t ix = ii[bid];
    int64_t jx = jj[bid];

    // Load poses into registers
    uint pose_i_idx = ix * 8;
    uint pose_j_idx = jx * 8;

    float3 ti = float3(Twc[pose_i_idx], Twc[pose_i_idx + 1], Twc[pose_i_idx + 2]);
    float4 qi = float4(Twc[pose_i_idx + 3], Twc[pose_i_idx + 4],
                       Twc[pose_i_idx + 5], Twc[pose_i_idx + 6]);
    float si = Twc[pose_i_idx + 7];

    float3 tj = float3(Twc[pose_j_idx], Twc[pose_j_idx + 1], Twc[pose_j_idx + 2]);
    float4 qj = float4(Twc[pose_j_idx + 3], Twc[pose_j_idx + 4],
                       Twc[pose_j_idx + 5], Twc[pose_j_idx + 6]);
    float sj = Twc[pose_j_idx + 7];

    // Compute relative pose
    float3 tij;
    float4 qij;
    float sij;
    rel_sim3(ti, qi, si, tj, qj, sj, tij, qij, sij);

    // Accumulate Hessian and gradient per thread
    const int h_dim = 14 * 15 / 2;  // Lower triangular
    float hij[h_dim];
    float vi[7], vj[7];

    for (int l = 0; l < h_dim; l++) hij[l] = 0.0f;
    for (int n = 0; n < 7; n++) { vi[n] = 0.0f; vj[n] = 0.0f; }

    // Process points assigned to this thread
    for (uint k = tid; k < num_points; k += THREADS) {
        uint edge_point_idx = bid * num_points + k;
        bool valid_match_ind = valid_match[edge_point_idx];
        int64_t ind_Xi = valid_match_ind ? idx_ii2jj[edge_point_idx] : 0;

        // Load points
        uint Xi_idx = ix * num_points * 3 + ind_Xi * 3;
        uint Xj_idx = jx * num_points * 3 + k * 3;

        float3 Xi = float3(Xs[Xi_idx], Xs[Xi_idx + 1], Xs[Xi_idx + 2]);
        float3 Xj = float3(Xs[Xj_idx], Xs[Xj_idx + 1], Xs[Xj_idx + 2]);

        // Normalize measurement
        float norm_i = length(Xi);
        float3 ri = Xi / norm_i;

        // Transform point
        float3 Xj_Ci = act_sim3(tij, qij, sij, Xj);
        float norm_j = length(Xj_Ci);
        float3 rj_Ci = Xj_Ci / norm_j;

        // Residuals
        float3 err_ray = rj_Ci - ri;
        float err_dist = norm_j - norm_i;

        // Load confidences
        float q_val = Q[edge_point_idx];
        float ci = Cs[ix * num_points + ind_Xi];
        float cj = Cs[jx * num_points + k];

        bool valid = valid_match_ind && (q_val > params.Q_thresh)
                     && (ci > params.C_thresh) && (cj > params.C_thresh);

        float conf_weight = q_val;
        float sqrt_w_ray = valid ? sigma_ray_inv * sqrt(conf_weight) : 0.0f;
        float sqrt_w_dist = valid ? sigma_dist_inv * sqrt(conf_weight) : 0.0f;

        // Huber weights
        float4 w;
        w.x = huber(sqrt_w_ray * err_ray.x) * sqrt_w_ray * sqrt_w_ray;
        w.y = huber(sqrt_w_ray * err_ray.y) * sqrt_w_ray * sqrt_w_ray;
        w.z = huber(sqrt_w_ray * err_ray.z) * sqrt_w_ray * sqrt_w_ray;
        w.w = huber(sqrt_w_dist * err_dist) * sqrt_w_dist * sqrt_w_dist;

        // Jacobians
        float Jx[14];
        thread float* Ji = &Jx[0];
        thread float* Jj = &Jx[7];

        float norm_j_inv = 1.0f / norm_j;
        float norm_j_inv3 = norm_j_inv / (norm_j * norm_j);

        // For each residual component, compute Jacobian and accumulate
        // rx component
        {
            float drx_dPx = norm_j_inv - Xj_Ci.x * Xj_Ci.x * norm_j_inv3;
            float drx_dPy = -Xj_Ci.x * Xj_Ci.y * norm_j_inv3;
            float drx_dPz = -Xj_Ci.x * Xj_Ci.z * norm_j_inv3;

            Ji[0] = drx_dPx; Ji[1] = drx_dPy; Ji[2] = drx_dPz;
            Ji[3] = 0.0f; Ji[4] = rj_Ci.z; Ji[5] = -rj_Ci.y; Ji[6] = 0.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.x * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.x * err_ray.x * Ji[n];
                vj[n] += w.x * err_ray.x * Jj[n];
            }
        }

        // ry component
        {
            float dry_dPx = -Xj_Ci.x * Xj_Ci.y * norm_j_inv3;
            float dry_dPy = norm_j_inv - Xj_Ci.y * Xj_Ci.y * norm_j_inv3;
            float dry_dPz = -Xj_Ci.y * Xj_Ci.z * norm_j_inv3;

            Ji[0] = dry_dPx; Ji[1] = dry_dPy; Ji[2] = dry_dPz;
            Ji[3] = -rj_Ci.z; Ji[4] = 0.0f; Ji[5] = rj_Ci.x; Ji[6] = 0.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.y * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.y * err_ray.y * Ji[n];
                vj[n] += w.y * err_ray.y * Jj[n];
            }
        }

        // rz component
        {
            float drz_dPx = -Xj_Ci.x * Xj_Ci.z * norm_j_inv3;
            float drz_dPy = -Xj_Ci.y * Xj_Ci.z * norm_j_inv3;
            float drz_dPz = norm_j_inv - Xj_Ci.z * Xj_Ci.z * norm_j_inv3;

            Ji[0] = drz_dPx; Ji[1] = drz_dPy; Ji[2] = drz_dPz;
            Ji[3] = rj_Ci.y; Ji[4] = -rj_Ci.x; Ji[5] = 0.0f; Ji[6] = 0.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.z * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.z * err_ray.z * Ji[n];
                vj[n] += w.z * err_ray.z * Jj[n];
            }
        }

        // dist component
        {
            Ji[0] = rj_Ci.x; Ji[1] = rj_Ci.y; Ji[2] = rj_Ci.z;
            Ji[3] = 0.0f; Ji[4] = 0.0f; Ji[5] = 0.0f; Ji[6] = norm_j;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.w * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.w * err_dist * Ji[n];
                vj[n] += w.w * err_dist * Jj[n];
            }
        }
    }

    // Block reduction for gradients
    for (int n = 0; n < 7; n++) {
        shared_data[tid] = vi[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction
        for (uint s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            gs[0 * params.num_edges * 7 + bid * 7 + n] = shared_data[0];
        }

        shared_data[tid] = vj[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            gs[1 * params.num_edges * 7 + bid * 7 + n] = shared_data[0];
        }
    }

    // Block reduction for Hessian
    int l = 0;
    for (int n = 0; n < 14; n++) {
        for (int m = 0; m <= n; m++) {
            shared_data[tid] = hij[l];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float val = shared_data[0];
                uint E = params.num_edges;

                if (n < 7 && m < 7) {
                    // Hs[0]: Hii block
                    Hs[0 * E * 49 + bid * 49 + n * 7 + m] = val;
                    Hs[0 * E * 49 + bid * 49 + m * 7 + n] = val;
                } else if (n >= 7 && m < 7) {
                    // Hs[1]: Hij block (transpose), Hs[2]: Hji block
                    Hs[1 * E * 49 + bid * 49 + m * 7 + (n - 7)] = val;
                    Hs[2 * E * 49 + bid * 49 + (n - 7) * 7 + m] = val;
                } else {
                    // Hs[3]: Hjj block
                    Hs[3 * E * 49 + bid * 49 + (n - 7) * 7 + (m - 7)] = val;
                    Hs[3 * E * 49 + bid * 49 + (m - 7) * 7 + (n - 7)] = val;
                }
            }

            l++;
        }
    }
}

// ============================================================================
// Calibrated Projection Kernel
// Uses pinhole camera model with pixel + log-depth residuals
// ============================================================================

kernel void calib_proj_kernel(
    device const float* Twc [[buffer(0)]],           // [num_poses, 8]
    device const float* Xs [[buffer(1)]],            // [num_poses, num_points, 3]
    device const float* Cs [[buffer(2)]],            // [num_poses, num_points, 1]
    device const float* K [[buffer(3)]],             // [3, 3] intrinsic matrix
    device const int64_t* ii [[buffer(4)]],          // [num_edges]
    device const int64_t* jj [[buffer(5)]],          // [num_edges]
    device const int64_t* idx_ii2jj [[buffer(6)]],   // [num_edges, num_points]
    device const bool* valid_match [[buffer(7)]],    // [num_edges, num_points, 1]
    device const float* Q [[buffer(8)]],             // [num_edges, num_points, 1]
    device float* Hs [[buffer(9)]],                  // [4, num_edges, 7, 7]
    device float* gs [[buffer(10)]],                 // [2, num_edges, 7]
    constant CalibParams& params [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    const uint num_points = params.num_points;
    const float sigma_pixel_inv = 1.0f / params.sigma_pixel;
    const float sigma_depth_inv = 1.0f / params.sigma_depth;

    // Load intrinsics
    float fx = K[0];  // K[0,0]
    float fy = K[4];  // K[1,1]
    float cx = K[2];  // K[0,2]
    float cy = K[5];  // K[1,2]

    // Get pose indices
    int64_t ix = ii[bid];
    int64_t jx = jj[bid];

    // Load poses
    uint pose_i_idx = ix * 8;
    uint pose_j_idx = jx * 8;

    float3 ti = float3(Twc[pose_i_idx], Twc[pose_i_idx + 1], Twc[pose_i_idx + 2]);
    float4 qi = float4(Twc[pose_i_idx + 3], Twc[pose_i_idx + 4],
                       Twc[pose_i_idx + 5], Twc[pose_i_idx + 6]);
    float si = Twc[pose_i_idx + 7];

    float3 tj = float3(Twc[pose_j_idx], Twc[pose_j_idx + 1], Twc[pose_j_idx + 2]);
    float4 qj = float4(Twc[pose_j_idx + 3], Twc[pose_j_idx + 4],
                       Twc[pose_j_idx + 5], Twc[pose_j_idx + 6]);
    float sj = Twc[pose_j_idx + 7];

    // Relative pose
    float3 tij;
    float4 qij;
    float sij;
    rel_sim3(ti, qi, si, tj, qj, sj, tij, qij, sij);

    // Accumulate Hessian and gradient
    const int h_dim = 14 * 15 / 2;
    float hij[h_dim];
    float vi[7], vj[7];

    for (int l = 0; l < h_dim; l++) hij[l] = 0.0f;
    for (int n = 0; n < 7; n++) { vi[n] = 0.0f; vj[n] = 0.0f; }

    for (uint k = tid; k < num_points; k += THREADS) {
        uint edge_point_idx = bid * num_points + k;
        bool valid_match_ind = valid_match[edge_point_idx];
        int64_t ind_Xi = valid_match_ind ? idx_ii2jj[edge_point_idx] : 0;

        // Load points
        uint Xi_idx = ix * num_points * 3 + ind_Xi * 3;
        uint Xj_idx = jx * num_points * 3 + k * 3;

        float3 Xi = float3(Xs[Xi_idx], Xs[Xi_idx + 1], Xs[Xi_idx + 2]);
        float3 Xj = float3(Xs[Xj_idx], Xs[Xj_idx + 1], Xs[Xj_idx + 2]);

        // Target pixel from index
        uint u_target = ind_Xi % params.width;
        uint v_target = ind_Xi / params.width;

        // Transform point
        float3 Xj_Ci = act_sim3(tij, qij, sij, Xj);

        // Depth validity
        bool valid_z = (Xj_Ci.z > params.z_eps) && (Xi.z > params.z_eps);
        float zj_inv = valid_z ? 1.0f / Xj_Ci.z : 0.0f;

        // Project
        float x_div_z = Xj_Ci.x * zj_inv;
        float y_div_z = Xj_Ci.y * zj_inv;
        float u = fx * x_div_z + cx;
        float v = fy * y_div_z + cy;

        // Check bounds
        bool valid_u = (u > params.pixel_border) && (u < params.width - 1 - params.pixel_border);
        bool valid_v = (v > params.pixel_border) && (v < params.height - 1 - params.pixel_border);

        // Residuals
        float err_u = u - float(u_target);
        float err_v = v - float(v_target);
        float err_z = valid_z ? log(Xj_Ci.z) - log(Xi.z) : 0.0f;

        // Weights
        float q_val = Q[edge_point_idx];
        float ci = Cs[ix * num_points + ind_Xi];
        float cj = Cs[jx * num_points + k];

        bool valid = valid_match_ind && (q_val > params.Q_thresh)
                     && (ci > params.C_thresh) && (cj > params.C_thresh)
                     && valid_u && valid_v && valid_z;

        float conf_weight = q_val;
        float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrt(conf_weight) : 0.0f;
        float sqrt_w_depth = valid ? sigma_depth_inv * sqrt(conf_weight) : 0.0f;

        float3 w;
        w.x = huber(sqrt_w_pixel * err_u) * sqrt_w_pixel * sqrt_w_pixel;
        w.y = huber(sqrt_w_pixel * err_v) * sqrt_w_pixel * sqrt_w_pixel;
        w.z = huber(sqrt_w_depth * err_z) * sqrt_w_depth * sqrt_w_depth;

        float Jx[14];
        thread float* Ji = &Jx[0];
        thread float* Jj = &Jx[7];

        // u residual Jacobian
        {
            Ji[0] = fx * zj_inv;
            Ji[1] = 0.0f;
            Ji[2] = -fx * x_div_z * zj_inv;
            Ji[3] = -fx * x_div_z * y_div_z;
            Ji[4] = fx * (1.0f + x_div_z * x_div_z);
            Ji[5] = -fx * y_div_z;
            Ji[6] = 0.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.x * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.x * err_u * Ji[n];
                vj[n] += w.x * err_u * Jj[n];
            }
        }

        // v residual Jacobian
        {
            Ji[0] = 0.0f;
            Ji[1] = fy * zj_inv;
            Ji[2] = -fy * y_div_z * zj_inv;
            Ji[3] = -fy * (1.0f + y_div_z * y_div_z);
            Ji[4] = fy * x_div_z * y_div_z;
            Ji[5] = fy * x_div_z;
            Ji[6] = 0.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.y * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.y * err_v * Ji[n];
                vj[n] += w.y * err_v * Jj[n];
            }
        }

        // z (log-depth) residual Jacobian
        {
            Ji[0] = 0.0f;
            Ji[1] = 0.0f;
            Ji[2] = zj_inv;
            Ji[3] = y_div_z;
            Ji[4] = -x_div_z;
            Ji[5] = 0.0f;
            Ji[6] = 1.0f;

            apply_sim3_adj_inv(ti, qi, si, Ji, Jj);
            for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

            int l = 0;
            for (int n = 0; n < 14; n++) {
                for (int m = 0; m <= n; m++) {
                    hij[l] += w.z * Jx[n] * Jx[m];
                    l++;
                }
            }
            for (int n = 0; n < 7; n++) {
                vi[n] += w.z * err_z * Ji[n];
                vj[n] += w.z * err_z * Jj[n];
            }
        }
    }

    // Block reduction for gradients
    for (int n = 0; n < 7; n++) {
        shared_data[tid] = vi[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            gs[0 * params.num_edges * 7 + bid * 7 + n] = shared_data[0];
        }

        shared_data[tid] = vj[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            gs[1 * params.num_edges * 7 + bid * 7 + n] = shared_data[0];
        }
    }

    // Block reduction for Hessian
    int l = 0;
    for (int n = 0; n < 14; n++) {
        for (int m = 0; m <= n; m++) {
            shared_data[tid] = hij[l];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                float val = shared_data[0];
                uint E = params.num_edges;

                if (n < 7 && m < 7) {
                    Hs[0 * E * 49 + bid * 49 + n * 7 + m] = val;
                    Hs[0 * E * 49 + bid * 49 + m * 7 + n] = val;
                } else if (n >= 7 && m < 7) {
                    Hs[1 * E * 49 + bid * 49 + m * 7 + (n - 7)] = val;
                    Hs[2 * E * 49 + bid * 49 + (n - 7) * 7 + m] = val;
                } else {
                    Hs[3 * E * 49 + bid * 49 + (n - 7) * 7 + (m - 7)] = val;
                    Hs[3 * E * 49 + bid * 49 + (m - 7) * 7 + (n - 7)] = val;
                }
            }

            l++;
        }
    }
}
