// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Metal shaders for Gauss-Newton optimization kernels.
 * Implements ray alignment and pose optimization for Apple Silicon GPU.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quaternion and Sim3 Operations
// ============================================================================

inline float4 quat_conj(float4 q) {
    return float4(-q.xyz, q.w);
}

inline float4 quat_mul(float4 q1, float4 q2) {
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

inline float3 quat_rotate(float4 q, float3 v) {
    float3 uv = 2.0f * cross(q.xyz, v);
    return v + q.w * uv + cross(q.xyz, uv);
}

inline float3 act_sim3(float3 t, float4 q, float s, float3 X) {
    return s * quat_rotate(q, X) + t;
}

// Sim3 exponential map
inline void exp_sim3(
    float3 tau,   // translation tangent
    float3 phi,   // rotation tangent
    float sigma,  // scale tangent
    thread float3& t_out,
    thread float4& q_out,
    thread float& s_out
) {
    float scale = exp(sigma);
    s_out = scale;

    // SO3 exponential
    float theta_sq = dot(phi, phi);
    float theta = sqrt(theta_sq);

    float imag, real;
    if (theta_sq < 1e-8f) {
        float theta_p4 = theta_sq * theta_sq;
        imag = 0.5f - theta_sq / 48.0f + theta_p4 / 3840.0f;
        real = 1.0f - theta_sq / 8.0f + theta_p4 / 384.0f;
    } else {
        imag = sin(0.5f * theta) / theta;
        real = cos(0.5f * theta);
    }
    q_out = float4(imag * phi, real);

    // Translation with W matrix
    float A, B, C;
    const float EPS = 1e-6f;

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

    // t = W @ tau
    t_out = C * tau;
    float3 cross1 = cross(phi, tau);
    t_out += A * cross1;
    float3 cross2 = cross(phi, cross1);
    t_out += B * cross2;
}

// ============================================================================
// Pose Retraction Kernel
// ============================================================================

struct PoseRetrParams {
    int n_poses;      // Total number of poses
    int num_fix;      // Number of fixed poses to skip
};

kernel void pose_retr_kernel(
    device float* poses [[buffer(0)]],           // [N, 8] in-place update
    device const float* delta [[buffer(1)]],     // [N-num_fix, 7] tangent vectors
    constant PoseRetrParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid indexes into delta array (0-indexed)
    // Actual pose index = gid + num_fix
    int n_opt = params.n_poses - params.num_fix;
    if (gid >= uint(n_opt)) return;

    int pose_k = gid + params.num_fix;

    // Read current pose: [tx, ty, tz, qx, qy, qz, qw, s]
    int pose_idx = pose_k * 8;
    float3 t = float3(poses[pose_idx], poses[pose_idx + 1], poses[pose_idx + 2]);
    float4 q = float4(poses[pose_idx + 3], poses[pose_idx + 4],
                      poses[pose_idx + 5], poses[pose_idx + 6]);
    float s = poses[pose_idx + 7];

    // Read delta: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z, sigma]
    // delta is indexed by gid (0 = first non-fixed pose)
    int delta_idx = gid * 7;
    float3 tau = float3(delta[delta_idx], delta[delta_idx + 1], delta[delta_idx + 2]);
    float3 phi = float3(delta[delta_idx + 3], delta[delta_idx + 4], delta[delta_idx + 5]);
    float sigma = delta[delta_idx + 6];

    // Compute exp(delta)
    float3 t_delta;
    float4 q_delta;
    float s_delta;
    exp_sim3(tau, phi, sigma, t_delta, q_delta, s_delta);

    // Compose: new_pose = exp(delta) @ pose
    float s_new = s_delta * s;
    float4 q_new = quat_mul(q_delta, q);
    float3 t_new = s_delta * quat_rotate(q_delta, t) + t_delta;

    // Normalize quaternion
    q_new = normalize(q_new);

    // Write back
    poses[pose_idx] = t_new.x;
    poses[pose_idx + 1] = t_new.y;
    poses[pose_idx + 2] = t_new.z;
    poses[pose_idx + 3] = q_new.x;
    poses[pose_idx + 4] = q_new.y;
    poses[pose_idx + 5] = q_new.z;
    poses[pose_idx + 6] = q_new.w;
    poses[pose_idx + 7] = s_new;
}

// ============================================================================
// Helper Functions
// ============================================================================

constant uint THREADS_GN = 256;
constant uint SIMD_SIZE_GN = 32;
constant float EPS_GN = 1e-6f;

inline float huber(float r) {
    const float r_abs = abs(r);
    return r_abs < 1.345f ? 1.0f : 1.345f / r_abs;
}

inline float squared_norm3(float3 v) {
    return dot(v, v);
}

inline float dot3(const thread float* t, const thread float* s) {
    return t[0]*s[0] + t[1]*s[1] + t[2]*s[2];
}

// Compute relative Sim3: Tij = Ti^-1 * Tj
inline void relSim3(
    float3 ti, float4 qi, float si,
    float3 tj, float4 qj, float sj,
    thread float3& tij, thread float4& qij, thread float& sij
) {
    // Inverse scale
    float si_inv = 1.0f / si;
    sij = si_inv * sj;

    // Relative rotation: qi^-1 * qj
    float4 qi_inv = quat_conj(qi);
    qij = quat_mul(qi_inv, qj);

    // Translation
    float3 diff = tj - ti;
    tij = si_inv * quat_rotate(qi_inv, diff);
}

// Apply Sim3 adjoint inverse (for Jacobian transformation)
// This is applying adj inv on the right to a row vector on the left
inline void apply_Sim3_adj_inv(
    float3 t, float4 q, float s,
    thread float* X,  // input: [7]
    thread float* Y   // output: [7]
) {
    const float s_inv = 1.0f / s;

    // Ra (rotate first 3 components)
    float3 a = float3(X[0], X[1], X[2]);
    float3 Ra = quat_rotate(q, a);

    // First component = s_inv * R * a
    Y[0] = s_inv * Ra.x;
    Y[1] = s_inv * Ra.y;
    Y[2] = s_inv * Ra.z;

    // Rb (rotate components 3-5)
    float3 b = float3(X[3], X[4], X[5]);
    float3 Rb = quat_rotate(q, b);

    // Second component = s_inv * [t]x * Ra + Rb
    Y[3] = Rb.x + s_inv * (t.y * Ra.z - t.z * Ra.y);
    Y[4] = Rb.y + s_inv * (t.z * Ra.x - t.x * Ra.z);
    Y[5] = Rb.z + s_inv * (t.x * Ra.y - t.y * Ra.x);

    // Third component = s_inv * t^T * Ra + c
    Y[6] = X[6] + s_inv * dot(t, Ra);
}

// ============================================================================
// Ray Alignment Kernel (Full Gauss-Newton)
// Computes Hessian and gradient for ray alignment optimization
// ============================================================================

struct RayAlignParams {
    int n_edges;
    int n_pts_per_frame;
    float sigma_ray;
    float sigma_dist;
    float C_thresh;
    float Q_thresh;
};

kernel void ray_align_kernel(
    device const float* poses [[buffer(0)]],       // [N_poses, 8]
    device const float* Xs [[buffer(1)]],          // [N_poses, M, 3]
    device const float* Cs [[buffer(2)]],          // [N_poses, M]
    device const int* ii [[buffer(3)]],            // [N_edges]
    device const int* jj [[buffer(4)]],            // [N_edges]
    device const int* idx_ii2jj [[buffer(5)]],     // [N_edges, M]
    device const bool* valid_match [[buffer(6)]],  // [N_edges, M]
    device const float* Q [[buffer(7)]],           // [N_edges, M]
    device float* Hs [[buffer(8)]],                // [4, N_edges, 7, 7] output
    device float* gs [[buffer(9)]],                // [2, N_edges, 7] output
    constant RayAlignParams& params [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]   // [THREADS_GN]
) {
    const int block_id = bid;
    const int thread_id = tid;
    const int num_points = params.n_pts_per_frame;

    if (block_id >= uint(params.n_edges)) return;

    int ix = ii[block_id];
    int jx = jj[block_id];

    // Shared memory for poses
    threadgroup float ti[3], tj[3], tij[3];
    threadgroup float qi[4], qj[4], qij[4];
    threadgroup float si[1], sj[1], sij[1];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load poses from global memory (cooperative loading)
    if (thread_id < 3) {
        ti[thread_id] = poses[ix * 8 + thread_id];
        tj[thread_id] = poses[jx * 8 + thread_id];
    }
    if (thread_id < 4) {
        qi[thread_id] = poses[ix * 8 + 3 + thread_id];
        qj[thread_id] = poses[jx * 8 + 3 + thread_id];
    }
    if (thread_id == 0) {
        si[0] = poses[ix * 8 + 7];
        sj[0] = poses[jx * 8 + 7];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate relative pose
    if (thread_id == 0) {
        float3 t_i = float3(ti[0], ti[1], ti[2]);
        float4 q_i = float4(qi[0], qi[1], qi[2], qi[3]);
        float3 t_j = float3(tj[0], tj[1], tj[2]);
        float4 q_j = float4(qj[0], qj[1], qj[2], qj[3]);

        float3 t_ij;
        float4 q_ij;
        float s_ij;
        relSim3(t_i, q_i, si[0], t_j, q_j, sj[0], t_ij, q_ij, s_ij);

        tij[0] = t_ij.x; tij[1] = t_ij.y; tij[2] = t_ij.z;
        qij[0] = q_ij.x; qij[1] = q_ij.y; qij[2] = q_ij.z; qij[3] = q_ij.w;
        sij[0] = s_ij;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Local accumulators
    const int h_dim = 14 * (14 + 1) / 2;  // 105 elements for lower triangular
    float hij[h_dim];
    float vi[7], vj[7];

    // Initialize to zero
    for (int l = 0; l < h_dim; l++) hij[l] = 0;
    for (int n = 0; n < 7; n++) { vi[n] = 0; vj[n] = 0; }

    const float sigma_ray_inv = 1.0f / params.sigma_ray;
    const float sigma_dist_inv = 1.0f / params.sigma_dist;

    // Process points assigned to this thread
    for (uint k = thread_id; k < uint(num_points); k += THREADS_GN) {
        // Get point validity
        const bool valid_match_ind = valid_match[block_id * num_points + k];
        const int ind_Xi = valid_match_ind ? idx_ii2jj[block_id * num_points + k] : 0;

        // Load points
        float3 Xi = float3(
            Xs[(ix * num_points + ind_Xi) * 3],
            Xs[(ix * num_points + ind_Xi) * 3 + 1],
            Xs[(ix * num_points + ind_Xi) * 3 + 2]
        );
        float3 Xj = float3(
            Xs[(jx * num_points + k) * 3],
            Xs[(jx * num_points + k) * 3 + 1],
            Xs[(jx * num_points + k) * 3 + 2]
        );

        // Normalize measurement point
        float norm2_i = squared_norm3(Xi);
        float norm1_i = sqrt(norm2_i);
        float norm1_i_inv = 1.0f / norm1_i;
        float3 ri = norm1_i_inv * Xi;

        // Transform Xj to camera i frame
        float3 t_ij = float3(tij[0], tij[1], tij[2]);
        float4 q_ij = float4(qij[0], qij[1], qij[2], qij[3]);
        float3 Xj_Ci = act_sim3(t_ij, q_ij, sij[0], Xj);

        // Normalize predicted point
        float norm2_j = squared_norm3(Xj_Ci);
        float norm1_j = sqrt(norm2_j);
        float norm1_j_inv = 1.0f / norm1_j;
        float3 rj_Ci = norm1_j_inv * Xj_Ci;

        // Compute errors
        float err[4];
        err[0] = rj_Ci.x - ri.x;
        err[1] = rj_Ci.y - ri.y;
        err[2] = rj_Ci.z - ri.z;
        err[3] = norm1_j - norm1_i;  // Distance

        // Compute weights
        float q = Q[block_id * num_points + k];
        float ci = Cs[ix * num_points + ind_Xi];
        float cj = Cs[jx * num_points + k];

        bool valid = valid_match_ind
            && (q > params.Q_thresh)
            && (ci > params.C_thresh)
            && (cj > params.C_thresh);

        float conf_weight = q;
        float sqrt_w_ray = valid ? sigma_ray_inv * sqrt(conf_weight) : 0;
        float sqrt_w_dist = valid ? sigma_dist_inv * sqrt(conf_weight) : 0;

        // Robust weights (Huber)
        float w[4];
        w[0] = huber(sqrt_w_ray * err[0]) * sqrt_w_ray * sqrt_w_ray;
        w[1] = huber(sqrt_w_ray * err[1]) * sqrt_w_ray * sqrt_w_ray;
        w[2] = huber(sqrt_w_ray * err[2]) * sqrt_w_ray * sqrt_w_ray;
        w[3] = huber(sqrt_w_dist * err[3]) * sqrt_w_dist * sqrt_w_dist;

        // Jacobian storage
        float Jx[14];
        float* Ji = &Jx[0];
        float* Jj = &Jx[7];

        // Load pose i for Jacobian transform
        float3 t_i = float3(ti[0], ti[1], ti[2]);
        float4 q_i = float4(qi[0], qi[1], qi[2], qi[3]);

        // Jacobian coefficients for ray normalization
        float norm3_j_inv = norm1_j_inv / norm2_j;
        float drx_dPx = norm1_j_inv - Xj_Ci.x * Xj_Ci.x * norm3_j_inv;
        float dry_dPy = norm1_j_inv - Xj_Ci.y * Xj_Ci.y * norm3_j_inv;
        float drz_dPz = norm1_j_inv - Xj_Ci.z * Xj_Ci.z * norm3_j_inv;
        float drx_dPy = -Xj_Ci.x * Xj_Ci.y * norm3_j_inv;
        float drx_dPz = -Xj_Ci.x * Xj_Ci.z * norm3_j_inv;
        float dry_dPz = -Xj_Ci.y * Xj_Ci.z * norm3_j_inv;

        // ---- rx coordinate ----
        Ji[0] = drx_dPx; Ji[1] = drx_dPy; Ji[2] = drx_dPz;
        Ji[3] = 0.0f; Ji[4] = rj_Ci.z; Ji[5] = -rj_Ci.y; Ji[6] = 0.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        int l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[0] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[0] * err[0] * Ji[n];
            vj[n] += w[0] * err[0] * Jj[n];
        }

        // ---- ry coordinate ----
        Ji[0] = drx_dPy; Ji[1] = dry_dPy; Ji[2] = dry_dPz;
        Ji[3] = -rj_Ci.z; Ji[4] = 0.0f; Ji[5] = rj_Ci.x; Ji[6] = 0.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[1] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[1] * err[1] * Ji[n];
            vj[n] += w[1] * err[1] * Jj[n];
        }

        // ---- rz coordinate ----
        Ji[0] = drx_dPz; Ji[1] = dry_dPz; Ji[2] = drz_dPz;
        Ji[3] = rj_Ci.y; Ji[4] = -rj_Ci.x; Ji[5] = 0.0f; Ji[6] = 0.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[2] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[2] * err[2] * Ji[n];
            vj[n] += w[2] * err[2] * Jj[n];
        }

        // ---- dist coordinate ----
        Ji[0] = rj_Ci.x; Ji[1] = rj_Ci.y; Ji[2] = rj_Ci.z;
        Ji[3] = 0.0f; Ji[4] = 0.0f; Ji[5] = 0.0f; Ji[6] = norm1_j;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[3] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[3] * err[3] * Ji[n];
            vj[n] += w[3] * err[3] * Jj[n];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block reduction for gradients
    const int pose_dim = 7;
    for (int n = 0; n < pose_dim; n++) {
        // Reduce vi[n]
        shared[thread_id] = vi[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction
        for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                shared[thread_id] += shared[thread_id + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_id == 0) {
            gs[0 * params.n_edges * pose_dim + block_id * pose_dim + n] = shared[0];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce vj[n]
        shared[thread_id] = vj[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                shared[thread_id] += shared[thread_id + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_id == 0) {
            gs[1 * params.n_edges * pose_dim + block_id * pose_dim + n] = shared[0];
        }
    }

    // Block reduction for Hessian blocks
    int l = 0;
    for (int n = 0; n < 14; n++) {
        for (int m = 0; m <= n; m++) {
            shared[thread_id] = hij[l];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
                if (thread_id < s) {
                    shared[thread_id] += shared[thread_id + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (thread_id == 0) {
                float val = shared[0];
                // Hs[4, n_edges, 7, 7] layout
                // Block 0: Hii, Block 1: Hij^T, Block 2: Hij, Block 3: Hjj
                if (n < 7 && m < 7) {
                    int idx = 0 * params.n_edges * 49 + block_id * 49 + n * 7 + m;
                    Hs[idx] = val;
                    if (n != m) {
                        Hs[0 * params.n_edges * 49 + block_id * 49 + m * 7 + n] = val;
                    }
                } else if (n >= 7 && m < 7) {
                    // Hij^T block (block 1) and Hij block (block 2)
                    Hs[1 * params.n_edges * 49 + block_id * 49 + m * 7 + (n - 7)] = val;
                    Hs[2 * params.n_edges * 49 + block_id * 49 + (n - 7) * 7 + m] = val;
                } else {
                    // Hjj block (block 3)
                    int idx = 3 * params.n_edges * 49 + block_id * 49 + (n - 7) * 7 + (m - 7);
                    Hs[idx] = val;
                    if (n != m) {
                        Hs[3 * params.n_edges * 49 + block_id * 49 + (m - 7) * 7 + (n - 7)] = val;
                    }
                }
            }
            l++;
        }
    }
}

// ============================================================================
// Calibrated Projection Kernel (Full Gauss-Newton)
// Computes Hessian and gradient for calibrated projection optimization
// ============================================================================

struct CalibProjParams {
    int n_edges;
    int n_pts_per_frame;
    int height;
    int width;
    int pixel_border;
    float z_eps;
    float sigma_pixel;
    float sigma_depth;
    float C_thresh;
    float Q_thresh;
};

kernel void calib_proj_kernel(
    device const float* poses [[buffer(0)]],       // [N_poses, 8]
    device const float* Xs [[buffer(1)]],          // [N_poses, M, 3]
    device const float* Cs [[buffer(2)]],          // [N_poses, M]
    device const float* K [[buffer(3)]],           // [3, 3]
    device const int* ii [[buffer(4)]],            // [N_edges]
    device const int* jj [[buffer(5)]],            // [N_edges]
    device const int* idx_ii2jj [[buffer(6)]],     // [N_edges, M]
    device const bool* valid_match [[buffer(7)]],  // [N_edges, M]
    device const float* Q [[buffer(8)]],           // [N_edges, M]
    device float* Hs [[buffer(9)]],                // [4, N_edges, 7, 7] output
    device float* gs [[buffer(10)]],               // [2, N_edges, 7] output
    constant CalibProjParams& params [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]]   // [THREADS_GN]
) {
    const int block_id = bid;
    const int thread_id = tid;
    const int num_points = params.n_pts_per_frame;

    if (block_id >= uint(params.n_edges)) return;

    int ix = ii[block_id];
    int jx = jj[block_id];

    // Shared memory for intrinsics and poses
    threadgroup float fx, fy, cx, cy;
    threadgroup float ti[3], tj[3], tij[3];
    threadgroup float qi[4], qj[4], qij[4];
    threadgroup float si[1], sj[1], sij[1];

    // Load intrinsics
    if (thread_id == 0) {
        fx = K[0];   // K[0,0]
        fy = K[4];   // K[1,1]
        cx = K[2];   // K[0,2]
        cy = K[5];   // K[1,2]
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load poses from global memory
    if (thread_id < 3) {
        ti[thread_id] = poses[ix * 8 + thread_id];
        tj[thread_id] = poses[jx * 8 + thread_id];
    }
    if (thread_id < 4) {
        qi[thread_id] = poses[ix * 8 + 3 + thread_id];
        qj[thread_id] = poses[jx * 8 + 3 + thread_id];
    }
    if (thread_id == 0) {
        si[0] = poses[ix * 8 + 7];
        sj[0] = poses[jx * 8 + 7];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate relative pose
    if (thread_id == 0) {
        float3 t_i = float3(ti[0], ti[1], ti[2]);
        float4 q_i = float4(qi[0], qi[1], qi[2], qi[3]);
        float3 t_j = float3(tj[0], tj[1], tj[2]);
        float4 q_j = float4(qj[0], qj[1], qj[2], qj[3]);

        float3 t_ij;
        float4 q_ij;
        float s_ij;
        relSim3(t_i, q_i, si[0], t_j, q_j, sj[0], t_ij, q_ij, s_ij);

        tij[0] = t_ij.x; tij[1] = t_ij.y; tij[2] = t_ij.z;
        qij[0] = q_ij.x; qij[1] = q_ij.y; qij[2] = q_ij.z; qij[3] = q_ij.w;
        sij[0] = s_ij;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Local accumulators
    const int h_dim = 14 * (14 + 1) / 2;
    float hij[h_dim];
    float vi[7], vj[7];

    for (int l = 0; l < h_dim; l++) hij[l] = 0;
    for (int n = 0; n < 7; n++) { vi[n] = 0; vj[n] = 0; }

    const float sigma_pixel_inv = 1.0f / params.sigma_pixel;
    const float sigma_depth_inv = 1.0f / params.sigma_depth;

    // Process points
    for (uint k = thread_id; k < uint(num_points); k += THREADS_GN) {
        const bool valid_match_ind = valid_match[block_id * num_points + k];
        const int ind_Xi = valid_match_ind ? idx_ii2jj[block_id * num_points + k] : 0;

        // Load points
        float3 Xi = float3(
            Xs[(ix * num_points + ind_Xi) * 3],
            Xs[(ix * num_points + ind_Xi) * 3 + 1],
            Xs[(ix * num_points + ind_Xi) * 3 + 2]
        );
        float3 Xj = float3(
            Xs[(jx * num_points + k) * 3],
            Xs[(jx * num_points + k) * 3 + 1],
            Xs[(jx * num_points + k) * 3 + 2]
        );

        // Get target pixel
        const int u_target = ind_Xi % params.width;
        const int v_target = ind_Xi / params.width;

        // Transform Xj to camera i frame
        float3 t_ij = float3(tij[0], tij[1], tij[2]);
        float4 q_ij = float4(qij[0], qij[1], qij[2], qij[3]);
        float3 Xj_Ci = act_sim3(t_ij, q_ij, sij[0], Xj);

        // Check depth validity
        bool valid_z = (Xj_Ci.z > params.z_eps) && (Xi.z > params.z_eps);
        float zj_inv = valid_z ? 1.0f / Xj_Ci.z : 0.0f;
        float zj_log = valid_z ? log(Xj_Ci.z) : 0.0f;
        float zi_log = valid_z ? log(Xi.z) : 0.0f;

        // Project point
        float x_div_z = Xj_Ci.x * zj_inv;
        float y_div_z = Xj_Ci.y * zj_inv;
        float u = fx * x_div_z + cx;
        float v = fy * y_div_z + cy;

        // Check projection validity
        bool valid_u = (u > params.pixel_border) && (u < params.width - 1 - params.pixel_border);
        bool valid_v = (v > params.pixel_border) && (v < params.height - 1 - params.pixel_border);

        // Errors
        float err[3];
        err[0] = u - float(u_target);
        err[1] = v - float(v_target);
        err[2] = zj_log - zi_log;

        // Compute weights
        float q = Q[block_id * num_points + k];
        float ci = Cs[ix * num_points + ind_Xi];
        float cj = Cs[jx * num_points + k];

        bool valid = valid_match_ind
            && (q > params.Q_thresh)
            && (ci > params.C_thresh)
            && (cj > params.C_thresh)
            && valid_u && valid_v && valid_z;

        float conf_weight = q;
        float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrt(conf_weight) : 0;
        float sqrt_w_depth = valid ? sigma_depth_inv * sqrt(conf_weight) : 0;

        float w[3];
        w[0] = huber(sqrt_w_pixel * err[0]) * sqrt_w_pixel * sqrt_w_pixel;
        w[1] = huber(sqrt_w_pixel * err[1]) * sqrt_w_pixel * sqrt_w_pixel;
        w[2] = huber(sqrt_w_depth * err[2]) * sqrt_w_depth * sqrt_w_depth;

        // Jacobians
        float Jx[14];
        float* Ji = &Jx[0];
        float* Jj = &Jx[7];

        float3 t_i = float3(ti[0], ti[1], ti[2]);
        float4 q_i = float4(qi[0], qi[1], qi[2], qi[3]);

        // ---- x coordinate (pixel u) ----
        Ji[0] = fx * zj_inv;
        Ji[1] = 0.0f;
        Ji[2] = -fx * x_div_z * zj_inv;
        Ji[3] = -fx * x_div_z * y_div_z;
        Ji[4] = fx * (1 + x_div_z * x_div_z);
        Ji[5] = -fx * y_div_z;
        Ji[6] = 0.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        int l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[0] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[0] * err[0] * Ji[n];
            vj[n] += w[0] * err[0] * Jj[n];
        }

        // ---- y coordinate (pixel v) ----
        Ji[0] = 0.0f;
        Ji[1] = fy * zj_inv;
        Ji[2] = -fy * y_div_z * zj_inv;
        Ji[3] = -fy * (1 + y_div_z * y_div_z);
        Ji[4] = fy * x_div_z * y_div_z;
        Ji[5] = fy * x_div_z;
        Ji[6] = 0.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[1] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[1] * err[1] * Ji[n];
            vj[n] += w[1] * err[1] * Jj[n];
        }

        // ---- z coordinate (log-depth) ----
        Ji[0] = 0.0f;
        Ji[1] = 0.0f;
        Ji[2] = zj_inv;
        Ji[3] = y_div_z;
        Ji[4] = -x_div_z;
        Ji[5] = 0.0f;
        Ji[6] = 1.0f;

        apply_Sim3_adj_inv(t_i, q_i, si[0], Ji, Jj);
        for (int n = 0; n < 7; n++) Ji[n] = -Jj[n];

        l = 0;
        for (int n = 0; n < 14; n++) {
            for (int m = 0; m <= n; m++) {
                hij[l] += w[2] * Jx[n] * Jx[m];
                l++;
            }
        }
        for (int n = 0; n < 7; n++) {
            vi[n] += w[2] * err[2] * Ji[n];
            vj[n] += w[2] * err[2] * Jj[n];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block reduction for gradients
    const int pose_dim = 7;
    for (int n = 0; n < pose_dim; n++) {
        shared[thread_id] = vi[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                shared[thread_id] += shared[thread_id + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_id == 0) {
            gs[0 * params.n_edges * pose_dim + block_id * pose_dim + n] = shared[0];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[thread_id] = vj[n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                shared[thread_id] += shared[thread_id + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_id == 0) {
            gs[1 * params.n_edges * pose_dim + block_id * pose_dim + n] = shared[0];
        }
    }

    // Block reduction for Hessian
    int l = 0;
    for (int n = 0; n < 14; n++) {
        for (int m = 0; m <= n; m++) {
            shared[thread_id] = hij[l];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = THREADS_GN / 2; s > 0; s >>= 1) {
                if (thread_id < s) {
                    shared[thread_id] += shared[thread_id + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (thread_id == 0) {
                float val = shared[0];
                if (n < 7 && m < 7) {
                    int idx = 0 * params.n_edges * 49 + block_id * 49 + n * 7 + m;
                    Hs[idx] = val;
                    if (n != m) {
                        Hs[0 * params.n_edges * 49 + block_id * 49 + m * 7 + n] = val;
                    }
                } else if (n >= 7 && m < 7) {
                    Hs[1 * params.n_edges * 49 + block_id * 49 + m * 7 + (n - 7)] = val;
                    Hs[2 * params.n_edges * 49 + block_id * 49 + (n - 7) * 7 + m] = val;
                } else {
                    int idx = 3 * params.n_edges * 49 + block_id * 49 + (n - 7) * 7 + (m - 7);
                    Hs[idx] = val;
                    if (n != m) {
                        Hs[3 * params.n_edges * 49 + block_id * 49 + (m - 7) * 7 + (n - 7)] = val;
                    }
                }
            }
            l++;
        }
    }
}
