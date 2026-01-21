// Copyright Delanoe Pirard / Aedelon. Apache 2.0
// Gauss-Newton calibrated projection kernel for MASt3R-SLAM on Apple Silicon
// Computes per-point residuals and Jacobians for 2D projection + log-depth

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

// Project 3D point to 2D using intrinsics K
inline float2 project(float3 X, float fx, float fy, float cx, float cy) {
    float z_inv = 1.0f / (X.z + EPS);
    return float2(
        fx * X.x * z_inv + cx,
        fy * X.y * z_inv + cy
    );
}

// Kernel: Compute per-point Jacobian contribution for calibrated projection
// Residuals: (u_proj - u_obs, v_proj - v_obs, log(z_proj) - log(z_obs))
kernel void gn_calib_jacobian_kernel(
    device const float* Twc [[buffer(0)]],           // [num_kf, 8]
    device const float* Xs [[buffer(1)]],            // [num_kf, num_pts, 3]
    device const float* Cs [[buffer(2)]],            // [num_kf, num_pts]
    device const int* ii [[buffer(3)]],              // [num_edges]
    device const int* jj [[buffer(4)]],              // [num_edges]
    device const int* idx_ii2jj [[buffer(5)]],       // [num_edges * num_pts]
    device const bool* valid_match [[buffer(6)]],    // [num_edges * num_pts]
    device const float* Q [[buffer(7)]],             // [num_edges * num_pts]
    device const float* K [[buffer(8)]],             // [4] = (fx, fy, cx, cy)
    device float* JtJ_i [[buffer(9)]],               // [num_edges * num_pts, 28]
    device float* Jtr_i [[buffer(10)]],              // [num_edges * num_pts, 7]
    device float* JtJ_j [[buffer(11)]],              // [num_edges * num_pts, 28]
    device float* Jtr_j [[buffer(12)]],              // [num_edges * num_pts, 7]
    device float* JtJ_ij [[buffer(13)]],             // [num_edges * num_pts, 49]
    device bool* valid_out [[buffer(14)]],           // [num_edges * num_pts]
    constant int& num_kf [[buffer(15)]],
    constant int& num_pts [[buffer(16)]],
    constant int& num_edges [[buffer(17)]],
    constant int& img_width [[buffer(18)]],
    constant int& img_height [[buffer(19)]],
    constant int& pixel_border [[buffer(20)]],
    constant float& z_eps [[buffer(21)]],
    constant float& sigma_pixel_inv [[buffer(22)]],
    constant float& sigma_depth_inv [[buffer(23)]],
    constant float& C_thresh [[buffer(24)]],
    constant float& Q_thresh [[buffer(25)]],
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

    // Load intrinsics
    float fx = K[0];
    float fy = K[1];
    float cx = K[2];
    float cy = K[3];

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

    // Transform Xj to frame i
    float3 Xj_Ci = sim3_act(tij, qij, sij, Xj);

    // Check depth validity
    if (Xj_Ci.z < z_eps || Xi.z < z_eps) return;

    // Project both points
    float2 proj_j = project(Xj_Ci, fx, fy, cx, cy);
    float2 proj_i = project(Xi, fx, fy, cx, cy);

    // Check projection bounds
    if (proj_j.x < pixel_border || proj_j.x >= img_width - pixel_border ||
        proj_j.y < pixel_border || proj_j.y >= img_height - pixel_border) {
        return;
    }

    // Residuals: projection error + log-depth error
    float err_u = (proj_j.x - proj_i.x) * sigma_pixel_inv;
    float err_v = (proj_j.y - proj_i.y) * sigma_pixel_inv;
    float err_z = (log(Xj_Ci.z) - log(Xi.z)) * sigma_depth_inv;

    // Weights
    float sqrt_w = sqrt(q_conf);
    float w[3];
    w[0] = huber_weight(sqrt_w * err_u) * sqrt_w * sqrt_w;
    w[1] = huber_weight(sqrt_w * err_v) * sqrt_w * sqrt_w;
    w[2] = huber_weight(sqrt_w * err_z) * sqrt_w * sqrt_w;

    // Jacobian of projection w.r.t. 3D point
    // d(u)/d(X) = [fx/z, 0, -fx*x/z^2]
    // d(v)/d(X) = [0, fy/z, -fy*y/z^2]
    // d(log(z))/d(X) = [0, 0, 1/z]
    float z_inv = 1.0f / Xj_Ci.z;
    float z_inv2 = z_inv * z_inv;

    float dproj_dX[3][3];
    dproj_dX[0][0] = fx * z_inv * sigma_pixel_inv;
    dproj_dX[0][1] = 0;
    dproj_dX[0][2] = -fx * Xj_Ci.x * z_inv2 * sigma_pixel_inv;
    dproj_dX[1][0] = 0;
    dproj_dX[1][1] = fy * z_inv * sigma_pixel_inv;
    dproj_dX[1][2] = -fy * Xj_Ci.y * z_inv2 * sigma_pixel_inv;
    dproj_dX[2][0] = 0;
    dproj_dX[2][1] = 0;
    dproj_dX[2][2] = z_inv * sigma_depth_inv;

    // Jacobian of X w.r.t. pose (same as ray kernel)
    float Ji_X[3][7], Jj_X[3][7];

    // Translation part
    Ji_X[0][0] = 1; Ji_X[0][1] = 0; Ji_X[0][2] = 0;
    Ji_X[1][0] = 0; Ji_X[1][1] = 1; Ji_X[1][2] = 0;
    Ji_X[2][0] = 0; Ji_X[2][1] = 0; Ji_X[2][2] = 1;

    // Rotation part [Xj_Ci]_x
    Ji_X[0][3] = 0;         Ji_X[0][4] = Xj_Ci.z;   Ji_X[0][5] = -Xj_Ci.y;
    Ji_X[1][3] = -Xj_Ci.z;  Ji_X[1][4] = 0;         Ji_X[1][5] = Xj_Ci.x;
    Ji_X[2][3] = Xj_Ci.y;   Ji_X[2][4] = -Xj_Ci.x;  Ji_X[2][5] = 0;

    // Scale part
    Ji_X[0][6] = Xj_Ci.x;
    Ji_X[1][6] = Xj_Ci.y;
    Ji_X[2][6] = Xj_Ci.z;

    // Compute Jj via adjoint
    float s_inv = 1.0f / si;
    float4 qi_inv = quat_inv(qi);

    for (int c = 0; c < 3; c++) {
        float3 Ji_t = float3(Ji_X[c][0], Ji_X[c][1], Ji_X[c][2]);
        float3 Ji_r = float3(Ji_X[c][3], Ji_X[c][4], Ji_X[c][5]);

        float3 Jj_t = s_inv * quat_rotate(qi_inv, Ji_t);
        float3 Jj_r = quat_rotate(qi_inv, Ji_r);

        Jj_X[c][0] = Jj_t.x; Jj_X[c][1] = Jj_t.y; Jj_X[c][2] = Jj_t.z;
        Jj_X[c][3] = Jj_r.x; Jj_X[c][4] = Jj_r.y; Jj_X[c][5] = Jj_r.z;
        Jj_X[c][6] = Ji_X[c][6];

        // Sign flip for Ji
        for (int k = 0; k < 7; k++) Ji_X[c][k] = -Jj_X[c][k];
    }

    // Chain rule: J_pose = dproj_dX @ dX_dpose
    float Ji[3][7], Jj[3][7];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 7; c++) {
            Ji[r][c] = 0;
            Jj[r][c] = 0;
            for (int k = 0; k < 3; k++) {
                Ji[r][c] += dproj_dX[r][k] * Ji_X[k][c];
                Jj[r][c] += dproj_dX[r][k] * Jj_X[k][c];
            }
        }
    }

    // Compute weighted JtJ and Jtr
    float e[3] = {err_u, err_v, err_z};

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
