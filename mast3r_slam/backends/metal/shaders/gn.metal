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
    int n_poses;
};

kernel void pose_retr_kernel(
    device float* poses [[buffer(0)]],           // [N, 8] in-place update
    device const float* delta [[buffer(1)]],     // [N, 7] tangent vectors
    constant PoseRetrParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.n_poses)) return;

    // Read current pose: [tx, ty, tz, qx, qy, qz, qw, s]
    int pose_idx = gid * 8;
    float3 t = float3(poses[pose_idx], poses[pose_idx + 1], poses[pose_idx + 2]);
    float4 q = float4(poses[pose_idx + 3], poses[pose_idx + 4],
                      poses[pose_idx + 5], poses[pose_idx + 6]);
    float s = poses[pose_idx + 7];

    // Read delta: [tau_x, tau_y, tau_z, phi_x, phi_y, phi_z, sigma]
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
// Ray Alignment Residual Kernel
// ============================================================================

struct RayAlignParams {
    int n_edges;
    int n_pts_per_frame;
    float sigma_ray;
    float sigma_dist;
    float C_thresh;
    float Q_thresh;
};

kernel void ray_align_residual_kernel(
    device const float* poses [[buffer(0)]],       // [N_poses, 8]
    device const float* points [[buffer(1)]],      // [N_poses, M, 3]
    device const float* confidences [[buffer(2)]], // [N_poses, M]
    device const int* ii [[buffer(3)]],            // [N_edges]
    device const int* jj [[buffer(4)]],            // [N_edges]
    device const int* idx_ii2jj [[buffer(5)]],     // [N_edges, M]
    device const bool* valid_match [[buffer(6)]],  // [N_edges, M]
    device const float* Q [[buffer(7)]],           // [N_edges, M]
    device float* residuals [[buffer(8)]],         // [N_edges, M, 3] output
    device float* weights [[buffer(9)]],           // [N_edges, M] output
    constant RayAlignParams& params [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int edge = gid.y;
    int pt = gid.x;

    if (edge >= params.n_edges || pt >= params.n_pts_per_frame) return;

    int M = params.n_pts_per_frame;
    int idx = edge * M + pt;

    // Check validity
    if (!valid_match[idx] || Q[idx] < params.Q_thresh) {
        residuals[idx * 3] = 0;
        residuals[idx * 3 + 1] = 0;
        residuals[idx * 3 + 2] = 0;
        weights[idx] = 0;
        return;
    }

    int i = ii[edge];
    int j = jj[edge];
    int pt_j = idx_ii2jj[idx];

    // Get poses
    int pose_i_idx = i * 8;
    float3 t_i = float3(poses[pose_i_idx], poses[pose_i_idx + 1], poses[pose_i_idx + 2]);
    float4 q_i = float4(poses[pose_i_idx + 3], poses[pose_i_idx + 4],
                        poses[pose_i_idx + 5], poses[pose_i_idx + 6]);
    float s_i = poses[pose_i_idx + 7];

    int pose_j_idx = j * 8;
    float3 t_j = float3(poses[pose_j_idx], poses[pose_j_idx + 1], poses[pose_j_idx + 2]);
    float4 q_j = float4(poses[pose_j_idx + 3], poses[pose_j_idx + 4],
                        poses[pose_j_idx + 5], poses[pose_j_idx + 6]);
    float s_j = poses[pose_j_idx + 7];

    // Get points
    int pt_i_idx = (i * M + pt) * 3;
    float3 X_i = float3(points[pt_i_idx], points[pt_i_idx + 1], points[pt_i_idx + 2]);

    int pt_j_idx = (j * M + pt_j) * 3;
    float3 X_j = float3(points[pt_j_idx], points[pt_j_idx + 1], points[pt_j_idx + 2]);

    // Check confidence
    float C_i = confidences[i * M + pt];
    float C_j = confidences[j * M + pt_j];
    if (C_i < params.C_thresh || C_j < params.C_thresh) {
        residuals[idx * 3] = 0;
        residuals[idx * 3 + 1] = 0;
        residuals[idx * 3 + 2] = 0;
        weights[idx] = 0;
        return;
    }

    // Transform points to world
    float3 pW_i = act_sim3(t_i, q_i, s_i, X_i);
    float3 pW_j = act_sim3(t_j, q_j, s_j, X_j);

    // Ray alignment residual
    float3 ray_i = normalize(pW_i - t_i);
    float3 ray_j = normalize(pW_j - t_j);

    float3 r_ray = ray_i - ray_j;
    float r_dist = length(pW_i) - length(pW_j);

    // Store residual (weighted)
    float w_ray = 1.0f / (params.sigma_ray * params.sigma_ray);
    float w_dist = 1.0f / (params.sigma_dist * params.sigma_dist);

    residuals[idx * 3] = r_ray.x * w_ray;
    residuals[idx * 3 + 1] = r_ray.y * w_ray;
    residuals[idx * 3 + 2] = r_ray.z * w_ray + r_dist * w_dist;
    weights[idx] = Q[idx] * min(C_i, C_j);
}
