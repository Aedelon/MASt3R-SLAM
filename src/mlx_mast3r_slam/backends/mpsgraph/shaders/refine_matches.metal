// Copyright Delanoe Pirard / Aedelon. Apache 2.0
// Match refinement kernel - local descriptor search on Apple Silicon

#include <metal_stdlib>
using namespace metal;

// Kernel: Refine matches by local descriptor search
// For each point, search in a local window and find best matching descriptor
kernel void refine_matches_kernel(
    device const float* D11 [[buffer(0)]],      // [B, H, W, D] - reference descriptors
    device const float* D21 [[buffer(1)]],      // [B, N, D] - query descriptors
    device const int* p1_in [[buffer(2)]],      // [B, N, 2] - initial positions (x, y)
    device int* p1_out [[buffer(3)]],           // [B, N, 2] - refined positions
    constant int& batch_size [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& width [[buffer(6)]],
    constant int& desc_dim [[buffer(7)]],
    constant int& num_pts [[buffer(8)]],
    constant int& radius [[buffer(9)]],
    constant int& dilation [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = batch_size * num_pts;
    if (tid >= uint(total)) return;

    int batch_idx = tid / num_pts;
    int pt_idx = tid % num_pts;

    // Load initial position
    int p_base = (batch_idx * num_pts + pt_idx) * 2;
    int cx = p1_in[p_base];
    int cy = p1_in[p_base + 1];

    // Load query descriptor
    int q_base = (batch_idx * num_pts + pt_idx) * desc_dim;

    float best_score = -1e10f;
    int best_x = cx;
    int best_y = cy;

    // Search in local window
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ny = cy + dy * dilation;
            int nx = cx + dx * dilation;

            // Bounds check
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            // Compute dot product score
            int ref_base = ((batch_idx * height + ny) * width + nx) * desc_dim;
            float score = 0.0f;

            for (int d = 0; d < desc_dim; d++) {
                score += D21[q_base + d] * D11[ref_base + d];
            }

            if (score > best_score) {
                best_score = score;
                best_x = nx;
                best_y = ny;
            }
        }
    }

    // Write refined position
    p1_out[p_base] = best_x;
    p1_out[p_base + 1] = best_y;
}

// Optimized kernel with shared memory for descriptor
kernel void refine_matches_opt_kernel(
    device const float* D11 [[buffer(0)]],
    device const float* D21 [[buffer(1)]],
    device const int* p1_in [[buffer(2)]],
    device int* p1_out [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& height [[buffer(5)]],
    constant int& width [[buffer(6)]],
    constant int& desc_dim [[buffer(7)]],
    constant int& num_pts [[buffer(8)]],
    constant int& radius [[buffer(9)]],
    constant int& dilation [[buffer(10)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Shared memory for query descriptors (up to 256 dims)
    threadgroup float query_desc[256];

    int total = batch_size * num_pts;
    if (tid >= uint(total)) return;

    int batch_idx = tid / num_pts;
    int pt_idx = tid % num_pts;

    // Load query descriptor to shared memory collaboratively
    int q_base = (batch_idx * num_pts + pt_idx) * desc_dim;

    // Each thread loads part of the descriptor
    int load_iters = (desc_dim + 31) / 32;  // Assuming max 32 threads per point
    for (int i = 0; i < load_iters; i++) {
        int d_idx = lid + i * 32;
        if (d_idx < desc_dim) {
            query_desc[d_idx] = D21[q_base + d_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load initial position
    int p_base = (batch_idx * num_pts + pt_idx) * 2;
    int cx = p1_in[p_base];
    int cy = p1_in[p_base + 1];

    float best_score = -1e10f;
    int best_x = cx;
    int best_y = cy;

    // Search in local window
    int window_size = 2 * radius + 1;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ny = cy + dy * dilation;
            int nx = cx + dx * dilation;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            int ref_base = ((batch_idx * height + ny) * width + nx) * desc_dim;
            float score = 0.0f;

            // Use SIMD for dot product (unrolled for common descriptor dims)
            int d = 0;
            for (; d + 4 <= desc_dim; d += 4) {
                score += query_desc[d] * D11[ref_base + d];
                score += query_desc[d + 1] * D11[ref_base + d + 1];
                score += query_desc[d + 2] * D11[ref_base + d + 2];
                score += query_desc[d + 3] * D11[ref_base + d + 3];
            }
            for (; d < desc_dim; d++) {
                score += query_desc[d] * D11[ref_base + d];
            }

            if (score > best_score) {
                best_score = score;
                best_x = nx;
                best_y = ny;
            }
        }
    }

    p1_out[p_base] = best_x;
    p1_out[p_base + 1] = best_y;
}

// Multi-scale refinement kernel
kernel void refine_matches_multiscale_kernel(
    device const float* D11 [[buffer(0)]],
    device const float* D21 [[buffer(1)]],
    device int* p1 [[buffer(2)]],              // In-place update
    constant int& batch_size [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& width [[buffer(5)]],
    constant int& desc_dim [[buffer(6)]],
    constant int& num_pts [[buffer(7)]],
    constant int& radius [[buffer(8)]],
    constant int& dilation [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = batch_size * num_pts;
    if (tid >= uint(total)) return;

    int batch_idx = tid / num_pts;
    int pt_idx = tid % num_pts;

    int p_base = (batch_idx * num_pts + pt_idx) * 2;
    int cx = p1[p_base];
    int cy = p1[p_base + 1];

    int q_base = (batch_idx * num_pts + pt_idx) * desc_dim;

    float best_score = -1e10f;
    int best_x = cx;
    int best_y = cy;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int ny = cy + dy * dilation;
            int nx = cx + dx * dilation;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            int ref_base = ((batch_idx * height + ny) * width + nx) * desc_dim;
            float score = 0.0f;

            for (int d = 0; d < desc_dim; d++) {
                score += D21[q_base + d] * D11[ref_base + d];
            }

            if (score > best_score) {
                best_score = score;
                best_x = nx;
                best_y = ny;
            }
        }
    }

    p1[p_base] = best_x;
    p1[p_base + 1] = best_y;
}
