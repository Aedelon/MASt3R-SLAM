// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Metal shaders for matching kernels.
 * Implements iter_proj and refine_matches for Apple Silicon GPU.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Helper functions
// ============================================================================

inline float3 normalize_safe(float3 v) {
    float norm = length(v);
    return v / max(norm, 1e-12f);
}

inline float clamp_val(float x, float min_val, float max_val) {
    return min(max(x, min_val), max_val);
}

// Bilinear interpolation for 3-component vector from [H, W, 9] tensor
inline void bilinear_sample_9ch(
    device const float* data,  // [H, W, 9]
    int H, int W,
    float u, float v,
    thread float3& ray,
    thread float3& grad_x,
    thread float3& grad_y
) {
    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);

    // Weights for bilinear interpolation
    float w00 = (1.0f - du) * (1.0f - dv);
    float w01 = (1.0f - du) * dv;
    float w10 = du * (1.0f - dv);
    float w11 = du * dv;

    // Clamp indices
    int u1 = min(u0 + 1, W - 1);
    int v1 = min(v0 + 1, H - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    // Sample 4 corners
    int idx00 = (v0 * W + u0) * 9;
    int idx01 = (v1 * W + u0) * 9;
    int idx10 = (v0 * W + u1) * 9;
    int idx11 = (v1 * W + u1) * 9;

    ray = float3(0);
    grad_x = float3(0);
    grad_y = float3(0);

    for (int c = 0; c < 3; c++) {
        ray[c] = w00 * data[idx00 + c] + w01 * data[idx01 + c] +
                 w10 * data[idx10 + c] + w11 * data[idx11 + c];
        grad_x[c] = w00 * data[idx00 + 3 + c] + w01 * data[idx01 + 3 + c] +
                    w10 * data[idx10 + 3 + c] + w11 * data[idx11 + 3 + c];
        grad_y[c] = w00 * data[idx00 + 6 + c] + w01 * data[idx01 + 6 + c] +
                    w10 * data[idx10 + 6 + c] + w11 * data[idx11 + 6 + c];
    }
}

// ============================================================================
// Iterative Projection Kernel
// ============================================================================

struct IterProjParams {
    int batch_size;
    int n_points;
    int H;
    int W;
    int max_iter;
    float lambda_init;
    float cost_thresh;
};

kernel void iter_proj_kernel(
    device const float* rays_img [[buffer(0)]],      // [B, H, W, 9]
    device const float* pts_3d [[buffer(1)]],        // [B, N, 3]
    device const float* p_init [[buffer(2)]],        // [B, N, 2]
    device float* p_out [[buffer(3)]],               // [B, N, 2]
    device bool* converged [[buffer(4)]],            // [B, N]
    constant IterProjParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    int B = params.batch_size;
    int N = params.n_points;
    int H = params.H;
    int W = params.W;

    // Get initial position
    int p_idx = (b * N + n) * 2;
    float u = p_init[p_idx];
    float v = p_init[p_idx + 1];

    // Clamp to valid range
    u = clamp_val(u, 1.0f, float(W - 2));
    v = clamp_val(v, 1.0f, float(H - 2));

    // Get target point
    int pt_idx = (b * N + n) * 3;
    float3 target = float3(pts_3d[pt_idx], pts_3d[pt_idx + 1], pts_3d[pt_idx + 2]);

    // Pointer to rays for this batch
    device const float* rays_batch = rays_img + b * H * W * 9;

    float lambda = params.lambda_init;
    bool conv = false;

    // Levenberg-Marquardt iterations
    for (int iter = 0; iter < params.max_iter; iter++) {
        float3 ray, grad_x, grad_y;
        bilinear_sample_9ch(rays_batch, H, W, u, v, ray, grad_x, grad_y);
        ray = normalize_safe(ray);

        float3 err = ray - target;
        float cost = dot(err, err);

        // 2x2 normal equations with damping
        float A00 = dot(grad_x, grad_x) + lambda;
        float A01 = dot(grad_x, grad_y);
        float A11 = dot(grad_y, grad_y) + lambda;
        float b0 = -dot(err, grad_x);
        float b1 = -dot(err, grad_y);

        // Solve 2x2 system
        float det_inv = 1.0f / (A00 * A11 - A01 * A01 + 1e-12f);
        float delta_u = det_inv * (A11 * b0 - A01 * b1);
        float delta_v = det_inv * (-A01 * b0 + A00 * b1);

        // Trial step
        float u_new = clamp_val(u + delta_u, 1.0f, float(W - 2));
        float v_new = clamp_val(v + delta_v, 1.0f, float(H - 2));

        // Check new cost
        float3 ray_new, gx_new, gy_new;
        bilinear_sample_9ch(rays_batch, H, W, u_new, v_new, ray_new, gx_new, gy_new);
        ray_new = normalize_safe(ray_new);
        float3 err_new = ray_new - target;
        float new_cost = dot(err_new, err_new);

        if (new_cost < cost) {
            u = u_new;
            v = v_new;
            lambda *= 0.1f;
            conv = new_cost < params.cost_thresh;
        } else {
            lambda *= 10.0f;
            conv = cost < params.cost_thresh;
        }
    }

    // Store results
    p_out[p_idx] = u;
    p_out[p_idx + 1] = v;
    converged[b * N + n] = conv;
}

// ============================================================================
// Refine Matches Kernel
// ============================================================================

struct RefineMatchesParams {
    int batch_size;
    int n_points;
    int H;
    int W;
    int fdim;
    int radius;
    int dilation_max;
};

kernel void refine_matches_kernel(
    device const float* D11 [[buffer(0)]],      // [B, H, W, F]
    device const float* D21 [[buffer(1)]],      // [B, N, F]
    device const int64_t* p1 [[buffer(2)]],     // [B, N, 2]
    device int64_t* p1_out [[buffer(3)]],       // [B, N, 2]
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    int B = params.batch_size;
    int N = params.n_points;
    int H = params.H;
    int W = params.W;
    int F = params.fdim;
    int radius = params.radius;
    int dilation_max = params.dilation_max;

    // Get current position
    int p_idx = (b * N + n) * 2;
    int64_t u0 = p1[p_idx];
    int64_t v0 = p1[p_idx + 1];

    // Pointer to query descriptor
    device const float* query = D21 + (b * N + n) * F;

    // Pointer to descriptor image
    device const float* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    // Multi-scale search (coarse to fine)
    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;

        for (int dv = -rd; dv <= rd; dv += d) {
            for (int du = -rd; du <= rd; du += d) {
                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                // Bounds check
                if (u < 0 || u >= W || v < 0 || v >= H) continue;

                // Compute dot product (descriptor correlation)
                device const float* desc = desc_img + (v * W + u) * F;
                float score = 0.0f;

                for (int f = 0; f < F; f++) {
                    score += query[f] * desc[f];
                }

                if (score > max_score) {
                    max_score = score;
                    u_best = u;
                    v_best = v;
                }
            }
        }

        // Update center for next scale
        u0 = u_best;
        v0 = v_best;
    }

    // Store result
    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}
