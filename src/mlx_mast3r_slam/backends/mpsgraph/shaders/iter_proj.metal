// Copyright Delanoe Pirard / Aedelon. Apache 2.0
// Iterative projection kernel for MASt3R-SLAM on Apple Silicon
// Uses Levenberg-Marquardt 2D optimization for point-to-ray matching

#include <metal_stdlib>
using namespace metal;

// Bilinear interpolation of ray image at given position
inline float3 bilinear_sample_ray(
    device const float* rays_img,  // [H, W, 9]
    int H, int W,
    float u, float v,
    thread float3& grad_x,
    thread float3& grad_y
) {
    // Clamp to valid range
    u = clamp(u, 0.0f, float(W - 1) - 0.001f);
    v = clamp(v, 0.0f, float(H - 1) - 0.001f);

    int x0 = int(floor(u));
    int y0 = int(floor(v));
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float fx = u - float(x0);
    float fy = v - float(y0);

    // Sample 4 corners
    int idx00 = (y0 * W + x0) * 9;
    int idx01 = (y1 * W + x0) * 9;
    int idx10 = (y0 * W + x1) * 9;
    int idx11 = (y1 * W + x1) * 9;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w10 = fx * (1.0f - fy);
    float w11 = fx * fy;

    // Interpolate rays [0:3]
    float3 ray;
    ray.x = w00 * rays_img[idx00 + 0] + w01 * rays_img[idx01 + 0] +
            w10 * rays_img[idx10 + 0] + w11 * rays_img[idx11 + 0];
    ray.y = w00 * rays_img[idx00 + 1] + w01 * rays_img[idx01 + 1] +
            w10 * rays_img[idx10 + 1] + w11 * rays_img[idx11 + 1];
    ray.z = w00 * rays_img[idx00 + 2] + w01 * rays_img[idx01 + 2] +
            w10 * rays_img[idx10 + 2] + w11 * rays_img[idx11 + 2];

    // Interpolate grad_x [3:6]
    grad_x.x = w00 * rays_img[idx00 + 3] + w01 * rays_img[idx01 + 3] +
               w10 * rays_img[idx10 + 3] + w11 * rays_img[idx11 + 3];
    grad_x.y = w00 * rays_img[idx00 + 4] + w01 * rays_img[idx01 + 4] +
               w10 * rays_img[idx10 + 4] + w11 * rays_img[idx11 + 4];
    grad_x.z = w00 * rays_img[idx00 + 5] + w01 * rays_img[idx01 + 5] +
               w10 * rays_img[idx10 + 5] + w11 * rays_img[idx11 + 5];

    // Interpolate grad_y [6:9]
    grad_y.x = w00 * rays_img[idx00 + 6] + w01 * rays_img[idx01 + 6] +
               w10 * rays_img[idx10 + 6] + w11 * rays_img[idx11 + 6];
    grad_y.y = w00 * rays_img[idx00 + 7] + w01 * rays_img[idx01 + 7] +
               w10 * rays_img[idx10 + 7] + w11 * rays_img[idx11 + 7];
    grad_y.z = w00 * rays_img[idx00 + 8] + w01 * rays_img[idx01 + 8] +
               w10 * rays_img[idx10 + 8] + w11 * rays_img[idx11 + 8];

    return ray;
}

// Solve 2x2 linear system: [a b; c d] * [x; y] = [e; f]
inline float2 solve_2x2(float a, float b, float c, float d, float e, float f) {
    float det = a * d - b * c;
    if (abs(det) < 1e-10f) {
        return float2(0.0f, 0.0f);
    }
    float inv_det = 1.0f / det;
    return float2(
        (d * e - b * f) * inv_det,
        (-c * e + a * f) * inv_det
    );
}

// Main iterative projection kernel
// Each thread handles one point
kernel void iter_proj_kernel(
    device const float* rays_img [[buffer(0)]],     // [B, H, W, 9]
    device const float* pts3d_norm [[buffer(1)]],   // [B, N, 3]
    device const float* p_init [[buffer(2)]],       // [B, N, 2]
    device float* p_out [[buffer(3)]],              // [B, N, 2]
    device bool* valid_out [[buffer(4)]],           // [B, N]
    constant int& B [[buffer(5)]],
    constant int& H [[buffer(6)]],
    constant int& W [[buffer(7)]],
    constant int& N [[buffer(8)]],
    constant int& max_iter [[buffer(9)]],
    constant float& lambda_init [[buffer(10)]],
    constant float& convergence_thresh [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    // Compute batch and point index
    int total = B * N;
    if (tid >= uint(total)) return;

    int b = tid / N;
    int n = tid % N;

    // Get initial position
    int p_idx = (b * N + n) * 2;
    float u = p_init[p_idx + 0];
    float v = p_init[p_idx + 1];

    // Get target ray
    int pt_idx = (b * N + n) * 3;
    float3 target_ray = float3(
        pts3d_norm[pt_idx + 0],
        pts3d_norm[pt_idx + 1],
        pts3d_norm[pt_idx + 2]
    );

    // Pointer to ray image for this batch
    device const float* rays_b = rays_img + b * H * W * 9;

    float lam = lambda_init;

    // Levenberg-Marquardt iterations
    for (int iter = 0; iter < max_iter; iter++) {
        // Sample ray and gradients at current position
        float3 grad_x, grad_y;
        float3 ray = bilinear_sample_ray(rays_b, H, W, u, v, grad_x, grad_y);

        // Residual: r = ray - target
        float3 r = ray - target_ray;

        // Build Jacobian J = [grad_x, grad_y] (3x2)
        // JtJ = J^T * J (2x2)
        float JtJ_00 = dot(grad_x, grad_x) + lam;
        float JtJ_01 = dot(grad_x, grad_y);
        float JtJ_11 = dot(grad_y, grad_y) + lam;

        // Jtr = J^T * r (2x1)
        float Jtr_0 = dot(grad_x, r);
        float Jtr_1 = dot(grad_y, r);

        // Solve: JtJ * delta = -Jtr
        float2 delta = solve_2x2(JtJ_00, JtJ_01, JtJ_01, JtJ_11, -Jtr_0, -Jtr_1);

        // Update position
        u += delta.x;
        v += delta.y;

        // Check convergence
        float delta_norm = length(delta);
        if (delta_norm < convergence_thresh) {
            break;
        }
    }

    // Clamp final position
    float u_final = clamp(u, 0.0f, float(W - 1));
    float v_final = clamp(v, 0.0f, float(H - 1));

    // Write output
    p_out[p_idx + 0] = u_final;
    p_out[p_idx + 1] = v_final;

    // Valid if within bounds
    valid_out[b * N + n] = (u >= 0.0f && u < float(W) && v >= 0.0f && v < float(H));
}

// Kernel for computing match validity based on 3D distance
kernel void validate_matches_kernel(
    device const float* X11 [[buffer(0)]],          // [B, H*W, 3]
    device const float* X21 [[buffer(1)]],          // [B, H*W, 3]
    device const int* idx [[buffer(2)]],            // [B, N] correspondence indices
    device const bool* valid_proj [[buffer(3)]],    // [B, N] projection validity
    device bool* valid_out [[buffer(4)]],           // [B, N] output validity
    constant int& B [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& HW [[buffer(7)]],
    constant float& dist_thresh [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = B * N;
    if (tid >= uint(total)) return;

    int b = tid / N;
    int n = tid % N;

    // Skip if projection invalid
    if (!valid_proj[b * N + n]) {
        valid_out[b * N + n] = false;
        return;
    }

    // Get correspondence index
    int corr_idx = idx[b * N + n];
    corr_idx = clamp(corr_idx, 0, HW - 1);

    // Sample X11 at correspondence
    int x11_idx = (b * HW + corr_idx) * 3;
    float3 x11 = float3(X11[x11_idx], X11[x11_idx + 1], X11[x11_idx + 2]);

    // Get X21 at current point
    int x21_idx = (b * N + n) * 3;
    float3 x21 = float3(X21[x21_idx], X21[x21_idx + 1], X21[x21_idx + 2]);

    // Compute 3D distance
    float dist = length(x11 - x21);

    // Valid if distance below threshold
    valid_out[b * N + n] = (dist < dist_thresh);
}
