// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Optimized Metal shaders for matching kernels.
 *
 * Optimizations:
 * - Threadgroup memory for shared data
 * - SIMD-group reductions for dot products
 * - Loop unrolling
 * - Half precision for descriptors
 * - Coalesced memory access patterns
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant int THREADGROUP_SIZE = 256;
constant int SIMD_SIZE = 32;

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

// Fast bilinear interpolation with prefetching
inline void bilinear_sample_9ch_fast(
    device const float* __restrict data,
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

    // Precompute weights
    float w00 = (1.0f - du) * (1.0f - dv);
    float w01 = (1.0f - du) * dv;
    float w10 = du * (1.0f - dv);
    float w11 = du * dv;

    // Clamp indices
    int u1 = min(u0 + 1, W - 1);
    int v1 = min(v0 + 1, H - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    // Prefetch all 4 corners (36 floats total)
    int idx00 = (v0 * W + u0) * 9;
    int idx01 = (v1 * W + u0) * 9;
    int idx10 = (v0 * W + u1) * 9;
    int idx11 = (v1 * W + u1) * 9;

    // Vectorized loads using packed_float3 (12 bytes, no padding)
    // 12 vector loads instead of 36 scalar loads
    device const packed_float3* p00 = (device const packed_float3*)(data + idx00);
    device const packed_float3* p01 = (device const packed_float3*)(data + idx01);
    device const packed_float3* p10 = (device const packed_float3*)(data + idx10);
    device const packed_float3* p11 = (device const packed_float3*)(data + idx11);

    // Load and interpolate ray (channel 0)
    ray = w00 * float3(p00[0]) + w01 * float3(p01[0]) +
          w10 * float3(p10[0]) + w11 * float3(p11[0]);

    // Load and interpolate grad_x (channel 1)
    grad_x = w00 * float3(p00[1]) + w01 * float3(p01[1]) +
             w10 * float3(p10[1]) + w11 * float3(p11[1]);

    // Load and interpolate grad_y (channel 2)
    grad_y = w00 * float3(p00[2]) + w01 * float3(p01[2]) +
             w10 * float3(p10[2]) + w11 * float3(p11[2]);
}

// ============================================================================
// Optimized Iterative Projection Kernel
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

kernel void iter_proj_kernel_optimized(
    device const float* __restrict rays_img [[buffer(0)]],
    device const float* __restrict pts_3d [[buffer(1)]],
    device const float* __restrict p_init [[buffer(2)]],
    device float* __restrict p_out [[buffer(3)]],
    device bool* __restrict converged [[buffer(4)]],
    constant IterProjParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;

    // Load initial position and target
    int p_idx = (b * N + n) * 2;
    float u = clamp_val(p_init[p_idx], 1.0f, float(W - 2));
    float v = clamp_val(p_init[p_idx + 1], 1.0f, float(H - 2));

    int pt_idx = (b * N + n) * 3;
    float3 target = float3(pts_3d[pt_idx], pts_3d[pt_idx + 1], pts_3d[pt_idx + 2]);

    device const float* rays_batch = rays_img + b * H * W * 9;

    float lambda = params.lambda_init;
    bool conv = false;

    // Unrolled LM iterations
    #pragma unroll 4
    for (int iter = 0; iter < params.max_iter; iter++) {
        float3 ray, grad_x, grad_y;
        bilinear_sample_9ch_fast(rays_batch, H, W, u, v, ray, grad_x, grad_y);
        ray = normalize_safe(ray);

        float3 err = ray - target;
        float cost = dot(err, err);

        // 2x2 normal equations
        float A00 = dot(grad_x, grad_x) + lambda;
        float A01 = dot(grad_x, grad_y);
        float A11 = dot(grad_y, grad_y) + lambda;
        float b0 = -dot(err, grad_x);
        float b1 = -dot(err, grad_y);

        // Solve 2x2 system (Cramer's rule)
        float det = A00 * A11 - A01 * A01;
        float det_inv = 1.0f / (det + 1e-12f);
        float delta_u = det_inv * (A11 * b0 - A01 * b1);
        float delta_v = det_inv * (A00 * b1 - A01 * b0);

        // Trial step
        float u_new = clamp_val(u + delta_u, 1.0f, float(W - 2));
        float v_new = clamp_val(v + delta_v, 1.0f, float(H - 2));

        // Evaluate new cost
        float3 ray_new, gx_new, gy_new;
        bilinear_sample_9ch_fast(rays_batch, H, W, u_new, v_new, ray_new, gx_new, gy_new);
        ray_new = normalize_safe(ray_new);
        float new_cost = dot(ray_new - target, ray_new - target);

        // Update based on cost reduction
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

    p_out[p_idx] = u;
    p_out[p_idx + 1] = v;
    converged[b * N + n] = conv;
}

// ============================================================================
// Optimized Refine Matches with SIMD-group Reductions
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

// SIMD-group optimized dot product for 24-dim descriptors
inline float simd_dot_24(
    device const float* __restrict a,
    device const float* __restrict b,
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each thread in SIMD-group loads different elements
    // For 24 dims with 32 lanes: each lane handles < 1 element on average
    float sum = 0.0f;

    // Vectorized loading and multiply-add
    if (simd_lane < 24) {
        sum = a[simd_lane] * b[simd_lane];
    }

    // SIMD-group reduction
    return simd_sum(sum);
}

// Optimized dot product with loop unrolling
inline float fast_dot(
    device const float* __restrict a,
    device const float* __restrict b,
    int dim
) {
    float sum = 0.0f;

    // Unroll by 8
    int k = 0;
    for (; k + 7 < dim; k += 8) {
        sum += a[k] * b[k];
        sum += a[k+1] * b[k+1];
        sum += a[k+2] * b[k+2];
        sum += a[k+3] * b[k+3];
        sum += a[k+4] * b[k+4];
        sum += a[k+5] * b[k+5];
        sum += a[k+6] * b[k+6];
        sum += a[k+7] * b[k+7];
    }

    // Handle remainder
    for (; k < dim; k++) {
        sum += a[k] * b[k];
    }

    return sum;
}

// Half precision version for descriptors
inline float fast_dot_half(
    device const half* __restrict a,
    device const half* __restrict b,
    int dim
) {
    float sum = 0.0f;

    // Use half4 for vectorized loading
    int k = 0;
    for (; k + 7 < dim; k += 8) {
        half4 a0 = *((device const half4*)(a + k));
        half4 a1 = *((device const half4*)(a + k + 4));
        half4 b0 = *((device const half4*)(b + k));
        half4 b1 = *((device const half4*)(b + k + 4));

        sum += dot(float4(a0), float4(b0));
        sum += dot(float4(a1), float4(b1));
    }

    for (; k < dim; k++) {
        sum += float(a[k]) * float(b[k]);
    }

    return sum;
}

kernel void refine_matches_kernel_optimized(
    device const float* __restrict D11 [[buffer(0)]],
    device const float* __restrict D21 [[buffer(1)]],
    device const int64_t* __restrict p1 [[buffer(2)]],
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    int p_idx = (b * N + n) * 2;
    int64_t u0 = p1[p_idx];
    int64_t v0 = p1[p_idx + 1];

    device const float* query = D21 + (b * N + n) * F;
    device const float* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    // Multi-scale search with early termination
    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;

        // Optimized search pattern: center first, then spiral outward
        for (int dv = -rd; dv <= rd; dv += d) {
            for (int du = -rd; du <= rd; du += d) {
                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                if (u < 0 || u >= W || v < 0 || v >= H) continue;

                device const float* desc = desc_img + (v * W + u) * F;
                float score = fast_dot(query, desc, F);

                if (score > max_score) {
                    max_score = score;
                    u_best = u;
                    v_best = v;
                }
            }
        }

        u0 = u_best;
        v0 = v_best;
    }

    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}

// Float input version - accepts float positions from iter_proj (no CPU sync needed)
kernel void refine_matches_kernel_from_float(
    device const float* __restrict D11 [[buffer(0)]],
    device const float* __restrict D21 [[buffer(1)]],
    device const float* __restrict p1_float [[buffer(2)]],  // Float positions!
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    // Read float positions and round to int
    int p_idx = (b * N + n) * 2;
    int64_t u0 = int64_t(round(p1_float[p_idx]));
    int64_t v0 = int64_t(round(p1_float[p_idx + 1]));

    // Clamp to valid range
    u0 = clamp(u0, int64_t(0), int64_t(W - 1));
    v0 = clamp(v0, int64_t(0), int64_t(H - 1));

    device const float* query = D21 + (b * N + n) * F;
    device const float* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;

        for (int dv = -rd; dv <= rd; dv += d) {
            for (int du = -rd; du <= rd; du += d) {
                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                if (u < 0 || u >= W || v < 0 || v >= H) continue;

                device const float* desc = desc_img + (v * W + u) * F;
                float score = fast_dot(query, desc, F);

                if (score > max_score) {
                    max_score = score;
                    u_best = u;
                    v_best = v;
                }
            }
        }

        u0 = u_best;
        v0 = v_best;
    }

    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}

// Half precision version for better memory bandwidth
kernel void refine_matches_kernel_half(
    device const half* __restrict D11 [[buffer(0)]],
    device const half* __restrict D21 [[buffer(1)]],
    device const int64_t* __restrict p1 [[buffer(2)]],
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    int p_idx = (b * N + n) * 2;
    int64_t u0 = p1[p_idx];
    int64_t v0 = p1[p_idx + 1];

    device const half* query = D21 + (b * N + n) * F;
    device const half* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;

        for (int dv = -rd; dv <= rd; dv += d) {
            for (int du = -rd; du <= rd; du += d) {
                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                if (u < 0 || u >= W || v < 0 || v >= H) continue;

                device const half* desc = desc_img + (v * W + u) * F;
                float score = fast_dot_half(query, desc, F);

                if (score > max_score) {
                    max_score = score;
                    u_best = u;
                    v_best = v;
                }
            }
        }

        u0 = u_best;
        v0 = v_best;
    }

    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}

// ============================================================================
// SIMD-Group Parallel Search - Multiple positions evaluated in parallel
// Each SIMD-group (32 threads) handles one point, threads evaluate different positions
// ============================================================================

kernel void refine_matches_kernel_simd(
    device const float* __restrict D11 [[buffer(0)]],
    device const float* __restrict D21 [[buffer(1)]],
    device const int64_t* __restrict p1 [[buffer(2)]],
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Each SIMD-group handles one point
    int b = gid.y;
    int point_idx = gid.x / SIMD_SIZE;  // SIMD-group index

    if (b >= params.batch_size || point_idx >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    int p_idx = (b * N + point_idx) * 2;
    int u_center = int(p1[p_idx]);
    int v_center = int(p1[p_idx + 1]);

    device const float* query = D21 + (b * N + point_idx) * F;
    device const float* desc_img = D11 + b * H * W * F;

    float thread_max_score = -INFINITY;
    int thread_u_best = u_center;
    int thread_v_best = v_center;

    // Multi-scale search with SIMD parallelism
    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;
        int search_size = 2 * rd / d + 1;  // e.g., 5 for radius=2, d=1
        int total_positions = search_size * search_size;

        // Each thread in SIMD-group evaluates different positions
        for (int pos = int(simd_lane); pos < total_positions; pos += SIMD_SIZE) {
            int dv = (pos / search_size) * d - rd;
            int du = (pos % search_size) * d - rd;

            int u = u_center + du;
            int v = v_center + dv;

            float score = -INFINITY;
            if (u >= 0 && u < W && v >= 0 && v < H) {
                device const float* desc = desc_img + (v * W + u) * F;
                score = fast_dot(query, desc, F);
            }

            if (score > thread_max_score) {
                thread_max_score = score;
                thread_u_best = u;
                thread_v_best = v;
            }
        }

        // SIMD-group reduction using O(1) intrinsics instead of O(log N) loop
        // Step 1: Find max score across all threads (O(1))
        float max_score = simd_max(thread_max_score);

        // Step 2: Find first lane that has the max score
        // Lanes without max use SIMD_SIZE (invalid), lanes with max use their lane id
        uint lane_with_max = (thread_max_score == max_score) ? simd_lane : SIMD_SIZE;
        lane_with_max = simd_min(lane_with_max);  // Get first lane with max (O(1))

        // Step 3: Broadcast coordinates from the winning lane (O(1))
        u_center = simd_shuffle(thread_u_best, ushort(lane_with_max));
        v_center = simd_shuffle(thread_v_best, ushort(lane_with_max));
    }

    // Only lane 0 writes result (u_center/v_center already have final best)
    if (simd_lane == 0) {
        p1_out[p_idx] = int64_t(u_center);
        p1_out[p_idx + 1] = int64_t(v_center);
    }
}

// ============================================================================
// Cooperative Loading with Threadgroup Memory
// Load descriptors into shared memory, then compute dot products
// ============================================================================

constant int TILE_SIZE = 8;  // Load 8 descriptors at a time
constant int DESC_DIM = 24;  // Descriptor dimension

kernel void refine_matches_kernel_cooperative(
    device const float* __restrict D11 [[buffer(0)]],
    device const float* __restrict D21 [[buffer(1)]],
    device const int64_t* __restrict p1 [[buffer(2)]],
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    threadgroup float* shared_query [[threadgroup(0)]],      // [DESC_DIM]
    threadgroup float* shared_descs [[threadgroup(1)]]       // [TILE_SIZE * DESC_DIM]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    // Load query into shared memory (collaborative)
    device const float* query_global = D21 + (b * N + n) * F;
    if (tid < uint(F)) {
        shared_query[tid] = query_global[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int p_idx = (b * N + n) * 2;
    int64_t u0 = p1[p_idx];
    int64_t v0 = p1[p_idx + 1];

    device const float* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;
        int search_size = 2 * rd / d + 1;
        int total_positions = search_size * search_size;

        // Process positions in tiles
        for (int tile_start = 0; tile_start < total_positions; tile_start += TILE_SIZE) {
            int tile_count = min(TILE_SIZE, total_positions - tile_start);

            // Load tile of descriptors into shared memory
            for (int t = 0; t < tile_count; t++) {
                int pos = tile_start + t;
                int dv = (pos / search_size) * d - rd;
                int du = (pos % search_size) * d - rd;

                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                // Load descriptor if valid
                if (u >= 0 && u < W && v >= 0 && v < H) {
                    device const float* desc = desc_img + (v * W + u) * F;
                    // Collaborative load: each thread loads part of descriptor
                    for (int k = int(tid); k < F; k += int(THREADGROUP_SIZE)) {
                        shared_descs[t * F + k] = desc[k];
                    }
                } else {
                    // Invalid position: zero out
                    for (int k = int(tid); k < F; k += int(THREADGROUP_SIZE)) {
                        shared_descs[t * F + k] = 0.0f;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute scores from shared memory
            for (int t = 0; t < tile_count; t++) {
                int pos = tile_start + t;
                int dv = (pos / search_size) * d - rd;
                int du = (pos % search_size) * d - rd;

                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                if (u >= 0 && u < W && v >= 0 && v < H) {
                    float score = 0.0f;
                    for (int k = 0; k < F; k++) {
                        score += shared_query[k] * shared_descs[t * F + k];
                    }

                    if (score > max_score) {
                        max_score = score;
                        u_best = u;
                        v_best = v;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        u0 = u_best;
        v0 = v_best;
    }

    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}

// ============================================================================
// Tiled version with threadgroup memory for better cache utilization
// ============================================================================

kernel void refine_matches_kernel_tiled(
    device const float* __restrict D11 [[buffer(0)]],
    device const float* __restrict D21 [[buffer(1)]],
    device const int64_t* __restrict p1 [[buffer(2)]],
    device int64_t* __restrict p1_out [[buffer(3)]],
    constant RefineMatchesParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    threadgroup float* shared_query [[threadgroup(0)]]
) {
    int b = gid.y;
    int n = gid.x;

    if (b >= params.batch_size || n >= params.n_points) return;

    const int N = params.n_points;
    const int H = params.H;
    const int W = params.W;
    const int F = params.fdim;
    const int radius = params.radius;
    const int dilation_max = params.dilation_max;

    // Load query descriptor into threadgroup memory
    device const float* query_global = D21 + (b * N + n) * F;

    // Collaborative loading of query into shared memory
    if (tid < uint(F)) {
        shared_query[tid] = query_global[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int p_idx = (b * N + n) * 2;
    int64_t u0 = p1[p_idx];
    int64_t v0 = p1[p_idx + 1];

    device const float* desc_img = D11 + b * H * W * F;

    float max_score = -INFINITY;
    int64_t u_best = u0;
    int64_t v_best = v0;

    for (int d = dilation_max; d > 0; d--) {
        int rd = radius * d;

        for (int dv = -rd; dv <= rd; dv += d) {
            for (int du = -rd; du <= rd; du += d) {
                int64_t u = u0 + du;
                int64_t v = v0 + dv;

                if (u < 0 || u >= W || v < 0 || v >= H) continue;

                device const float* desc = desc_img + (v * W + u) * F;

                // Use shared query for dot product
                float score = 0.0f;
                for (int k = 0; k < F; k++) {
                    score += shared_query[k] * desc[k];
                }

                if (score > max_score) {
                    max_score = score;
                    u_best = u;
                    v_best = v;
                }
            }
        }

        u0 = u_best;
        v0 = v_best;
    }

    p1_out[p_idx] = u_best;
    p1_out[p_idx + 1] = v_best;
}
