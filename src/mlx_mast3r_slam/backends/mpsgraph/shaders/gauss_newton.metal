// Copyright Delanoe Pirard / Aedelon. Apache 2.0
// Gauss-Newton ray alignment kernel for MASt3R-SLAM on Apple Silicon
// Computes per-point residuals and Jacobians (accumulation done on CPU)

#include <metal_stdlib>
using namespace metal;

#define EPS 1e-6f

// Huber weight function
inline float huber_weight(float r, float k = 1.345f) {
    float r_abs = abs(r);
    return r_abs < k ? 1.0f : k / r_abs;
}

// Quaternion multiplication: q1 * q2
inline float4 quat_mul(float4 q1, float4 q2) {
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Quaternion inverse
inline float4 quat_inv(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

// Rotate vector by quaternion
inline float3 quat_rotate(float4 q, float3 v) {
    float3 qv = float3(q.x, q.y, q.z);
    float3 uv = 2.0f * cross(qv, v);
    return v + q.w * uv + cross(qv, uv);
}

// Compute relative Sim3: Tij = Ti^-1 * Tj
inline void rel_sim3(
    float3 ti, float4 qi, float si,
    float3 tj, float4 qj, float sj,
    thread float3& tij, thread float4& qij, thread float& sij
) {
    float si_inv = 1.0f / si;
    sij = si_inv * sj;

    float4 qi_inv = quat_inv(qi);
    qij = quat_mul(qi_inv, qj);

    tij = tj - ti;
    tij = quat_rotate(qi_inv, tij);
    tij *= si_inv;
}

// Apply Sim3 to point
inline float3 sim3_act(float3 t, float4 q, float s, float3 X) {
    float3 Y = quat_rotate(q, X);
    Y *= s;
    Y += t;
    return Y;
}

// Kernel: Compute per-point Jacobian contribution
// Output: weighted JtJ (7x7 upper triangular = 28 floats) and Jtr (7 floats) per point
// Total output per valid point: 28 + 7 + 28 + 7 + 49 = 119 floats for both poses + cross term
kernel void gn_jacobian_kernel(
    device const float* Twc [[buffer(0)]],           // [num_kf, 8]
    device const float* Xs [[buffer(1)]],            // [num_kf, num_pts, 3]
    device const float* Cs [[buffer(2)]],            // [num_kf, num_pts]
    device const int* ii [[buffer(3)]],              // [num_edges]
    device const int* jj [[buffer(4)]],              // [num_edges]
    device const int* idx_ii2jj [[buffer(5)]],       // [num_edges * num_pts]
    device const bool* valid_match [[buffer(6)]],    // [num_edges * num_pts]
    device const float* Q [[buffer(7)]],             // [num_edges * num_pts]
    device float* JtJ_i [[buffer(8)]],               // [num_edges * num_pts, 28] - upper tri of Hii
    device float* Jtr_i [[buffer(9)]],               // [num_edges * num_pts, 7]
    device float* JtJ_j [[buffer(10)]],              // [num_edges * num_pts, 28]
    device float* Jtr_j [[buffer(11)]],              // [num_edges * num_pts, 7]
    device float* JtJ_ij [[buffer(12)]],             // [num_edges * num_pts, 49] - full Hij
    device bool* valid_out [[buffer(13)]],           // [num_edges * num_pts]
    constant int& num_kf [[buffer(14)]],
    constant int& num_pts [[buffer(15)]],
    constant int& num_edges [[buffer(16)]],
    constant float& sigma_inv [[buffer(17)]],
    constant float& C_thresh [[buffer(18)]],
    constant float& Q_thresh [[buffer(19)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = num_edges * num_pts;
    if (tid >= uint(total)) return;

    int edge_idx = tid / num_pts;
    int pt_idx = tid % num_pts;

    // Initialize output as invalid
    valid_out[tid] = false;

    // Check validity
    if (!valid_match[tid]) return;

    float q_conf = Q[tid];
    if (q_conf <= Q_thresh) return;

    int ix = ii[edge_idx];
    int jx = jj[edge_idx];

    int corr_idx = idx_ii2jj[tid];
    if (corr_idx < 0 || corr_idx >= num_pts) return;

    // Check confidence
    float ci = Cs[ix * num_pts + corr_idx];
    float cj = Cs[jx * num_pts + pt_idx];
    if (ci <= C_thresh || cj <= C_thresh) return;

    // Load poses
    int pose_i = ix * 8;
    int pose_j = jx * 8;

    float3 ti = float3(Twc[pose_i], Twc[pose_i + 1], Twc[pose_i + 2]);
    float4 qi = float4(Twc[pose_i + 3], Twc[pose_i + 4], Twc[pose_i + 5], Twc[pose_i + 6]);
    float si = Twc[pose_i + 7];

    float3 tj = float3(Twc[pose_j], Twc[pose_j + 1], Twc[pose_j + 2]);
    float4 qj = float4(Twc[pose_j + 3], Twc[pose_j + 4], Twc[pose_j + 5], Twc[pose_j + 6]);
    float sj = Twc[pose_j + 7];

    // Relative transformation
    float3 tij; float4 qij; float sij;
    rel_sim3(ti, qi, si, tj, qj, sj, tij, qij, sij);

    // Load points
    int Xi_idx = ix * num_pts * 3 + corr_idx * 3;
    int Xj_idx = jx * num_pts * 3 + pt_idx * 3;

    float3 Xi = float3(Xs[Xi_idx], Xs[Xi_idx + 1], Xs[Xi_idx + 2]);
    float3 Xj = float3(Xs[Xj_idx], Xs[Xj_idx + 1], Xs[Xj_idx + 2]);

    // Transform
    float3 Xj_Ci = sim3_act(tij, qij, sij, Xj);

    // Residual
    float3 err = Xj_Ci - Xi;

    // Weight
    float sqrt_w = sigma_inv * sqrt(q_conf);
    float w[3];
    w[0] = huber_weight(sqrt_w * err.x) * sqrt_w * sqrt_w;
    w[1] = huber_weight(sqrt_w * err.y) * sqrt_w * sqrt_w;
    w[2] = huber_weight(sqrt_w * err.z) * sqrt_w * sqrt_w;

    // Build Jacobians
    float Ji[3][7], Jj[3][7];

    // Translation part
    Ji[0][0] = 1; Ji[0][1] = 0; Ji[0][2] = 0;
    Ji[1][0] = 0; Ji[1][1] = 1; Ji[1][2] = 0;
    Ji[2][0] = 0; Ji[2][1] = 0; Ji[2][2] = 1;

    // Rotation part [Xj_Ci]_x
    Ji[0][3] = 0;        Ji[0][4] = Xj_Ci.z;  Ji[0][5] = -Xj_Ci.y;
    Ji[1][3] = -Xj_Ci.z; Ji[1][4] = 0;        Ji[1][5] = Xj_Ci.x;
    Ji[2][3] = Xj_Ci.y;  Ji[2][4] = -Xj_Ci.x; Ji[2][5] = 0;

    // Scale part
    Ji[0][6] = Xj_Ci.x;
    Ji[1][6] = Xj_Ci.y;
    Ji[2][6] = Xj_Ci.z;

    // Compute Jj via adjoint
    float s_inv = 1.0f / si;
    float4 qi_inv = quat_inv(qi);

    for (int c = 0; c < 3; c++) {
        float3 Ji_t = float3(Ji[c][0], Ji[c][1], Ji[c][2]);
        float3 Ji_r = float3(Ji[c][3], Ji[c][4], Ji[c][5]);

        float3 Jj_t = s_inv * quat_rotate(qi_inv, Ji_t);
        float3 Jj_r = quat_rotate(qi_inv, Ji_r);

        Jj[c][0] = Jj_t.x; Jj[c][1] = Jj_t.y; Jj[c][2] = Jj_t.z;
        Jj[c][3] = Jj_r.x; Jj[c][4] = Jj_r.y; Jj[c][5] = Jj_r.z;
        Jj[c][6] = Ji[c][6];

        // Sign flip
        for (int k = 0; k < 7; k++) Ji[c][k] = -Jj[c][k];
    }

    // Compute weighted JtJ and Jtr
    float e[3] = {err.x, err.y, err.z};

    // JtJ_i: upper triangular (28 values)
    int out_i = tid * 28;
    int idx = 0;
    for (int m = 0; m < 7; m++) {
        for (int n = m; n < 7; n++) {
            float val = 0;
            for (int c = 0; c < 3; c++) {
                val += w[c] * Ji[c][m] * Ji[c][n];
            }
            JtJ_i[out_i + idx] = val;
            idx++;
        }
    }

    // Jtr_i
    int out_gi = tid * 7;
    for (int m = 0; m < 7; m++) {
        float val = 0;
        for (int c = 0; c < 3; c++) {
            val += w[c] * Ji[c][m] * e[c];
        }
        Jtr_i[out_gi + m] = val;
    }

    // JtJ_j: upper triangular
    int out_j = tid * 28;
    idx = 0;
    for (int m = 0; m < 7; m++) {
        for (int n = m; n < 7; n++) {
            float val = 0;
            for (int c = 0; c < 3; c++) {
                val += w[c] * Jj[c][m] * Jj[c][n];
            }
            JtJ_j[out_j + idx] = val;
            idx++;
        }
    }

    // Jtr_j
    int out_gj = tid * 7;
    for (int m = 0; m < 7; m++) {
        float val = 0;
        for (int c = 0; c < 3; c++) {
            val += w[c] * Jj[c][m] * e[c];
        }
        Jtr_j[out_gj + m] = val;
    }

    // JtJ_ij: full 7x7 cross term
    int out_ij = tid * 49;
    for (int m = 0; m < 7; m++) {
        for (int n = 0; n < 7; n++) {
            float val = 0;
            for (int c = 0; c < 3; c++) {
                val += w[c] * Ji[c][m] * Jj[c][n];
            }
            JtJ_ij[out_ij + m * 7 + n] = val;
        }
    }

    valid_out[tid] = true;
}

// Kernel: Apply pose update (retraction)
kernel void pose_update_kernel(
    device float* Twc [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device const int* free_kf_idx [[buffer(2)]],
    constant int& num_free [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(num_free)) return;

    int kf_idx = free_kf_idx[tid];
    int pose_base = kf_idx * 8;
    int dx_base = tid * 7;

    float3 t = float3(Twc[pose_base], Twc[pose_base + 1], Twc[pose_base + 2]);
    float4 q = float4(Twc[pose_base + 3], Twc[pose_base + 4], Twc[pose_base + 5], Twc[pose_base + 6]);
    float s = Twc[pose_base + 7];

    float3 tau = float3(dx[dx_base], dx[dx_base + 1], dx[dx_base + 2]);
    float3 omega = float3(dx[dx_base + 3], dx[dx_base + 4], dx[dx_base + 5]);
    float sigma = dx[dx_base + 6];

    // exp(omega)
    float theta_sq = dot(omega, omega);
    float theta = sqrt(theta_sq + EPS);
    float half_theta = 0.5f * theta;

    float imag = theta_sq < EPS ? 0.5f - theta_sq / 48.0f : sin(half_theta) / theta;
    float real = theta_sq < EPS ? 1.0f - theta_sq / 8.0f : cos(half_theta);

    float4 dq = float4(imag * omega.x, imag * omega.y, imag * omega.z, real);
    float ds = exp(sigma);
    float3 dt = tau;

    // Compose
    float4 q_new = quat_mul(dq, q);
    float3 t_new = quat_rotate(dq, t) * ds + dt;
    float s_new = ds * s;

    Twc[pose_base + 0] = t_new.x;
    Twc[pose_base + 1] = t_new.y;
    Twc[pose_base + 2] = t_new.z;
    Twc[pose_base + 3] = q_new.x;
    Twc[pose_base + 4] = q_new.y;
    Twc[pose_base + 5] = q_new.z;
    Twc[pose_base + 6] = q_new.w;
    Twc[pose_base + 7] = s_new;
}
