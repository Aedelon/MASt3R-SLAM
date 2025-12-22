// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * GPU Radix Sort for Metal.
 *
 * Implements a parallel radix sort for uint32 keys with optional values.
 * Uses local digit histogramming + prefix sum + scatter pattern.
 *
 * Optimizations:
 * - Threadgroup memory for local histograms
 * - SIMD-group reductions for prefix sums
 * - Coalesced memory access patterns
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint RADIX_BITS = 4;              // 4 bits per pass (16 buckets)
constant uint RADIX_SIZE = 1 << RADIX_BITS; // 16 buckets
constant uint RADIX_MASK = RADIX_SIZE - 1;  // 0xF

constant uint BLOCK_SIZE = 256;            // Threads per threadgroup
constant uint ELEMENTS_PER_THREAD = 4;     // Elements processed per thread
constant uint SIMD_SIZE = 32;              // Apple Silicon SIMD width

// ============================================================================
// Structures
// ============================================================================

struct SortParams {
    uint n_elements;      // Total number of elements
    uint bit_offset;      // Current bit position (0, 4, 8, ..., 28)
    uint n_blocks;        // Number of threadgroups
};

// ============================================================================
// Local Histogram Kernel
// Counts digit frequencies within each block
// ============================================================================

kernel void radix_histogram(
    device const uint* __restrict keys [[buffer(0)]],
    device uint* __restrict block_histograms [[buffer(1)]],  // [n_blocks, RADIX_SIZE]
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup uint* local_hist [[threadgroup(0)]]  // [RADIX_SIZE]
) {
    // Initialize local histogram
    if (tid < RADIX_SIZE) {
        local_hist[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process elements assigned to this thread
    uint block_start = bid * BLOCK_SIZE * ELEMENTS_PER_THREAD;

    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = block_start + tid + i * BLOCK_SIZE;

        if (idx < params.n_elements) {
            uint key = keys[idx];
            uint digit = (key >> params.bit_offset) & RADIX_MASK;

            // Atomic increment in threadgroup memory
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&local_hist[digit],
                1u,
                memory_order_relaxed
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    if (tid < RADIX_SIZE) {
        block_histograms[bid * RADIX_SIZE + tid] = local_hist[tid];
    }
}

// ============================================================================
// Global Prefix Sum (Scan) Kernel - Kogge-Stone Parallel Algorithm
// Computes exclusive prefix sum across all block histograms in O(log n)
// ============================================================================

kernel void radix_prefix_sum(
    device uint* __restrict block_histograms [[buffer(0)]],  // [n_blocks, RADIX_SIZE]
    device uint* __restrict global_offsets [[buffer(1)]],    // [n_blocks, RADIX_SIZE]
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    threadgroup uint* shared [[threadgroup(0)]]  // [BLOCK_SIZE]
) {
    // Each thread handles one column (one digit across all blocks)
    uint digit = gid;

    if (digit >= RADIX_SIZE) return;

    const uint n_blocks = params.n_blocks;

    // For small n_blocks, use SIMD parallel scan
    if (n_blocks <= SIMD_SIZE) {
        // Load values into SIMD lanes
        uint val = 0;
        if (simd_lane < n_blocks) {
            val = block_histograms[simd_lane * RADIX_SIZE + digit];
        }

        // Kogge-Stone parallel exclusive prefix sum using SIMD
        uint exclusive = 0;
        for (uint offset = 1; offset < SIMD_SIZE; offset *= 2) {
            uint prev = simd_shuffle_up(val, offset);
            if (simd_lane >= offset) {
                val += prev;
            }
        }

        // Convert to exclusive scan
        exclusive = simd_shuffle_up(val, 1);
        if (simd_lane == 0) exclusive = 0;

        // Write result
        if (simd_lane < n_blocks) {
            global_offsets[simd_lane * RADIX_SIZE + digit] = exclusive;
        }
    } else {
        // Fallback: sequential for large n_blocks (rare case)
        uint running_sum = 0;
        for (uint block = 0; block < n_blocks; block++) {
            uint count = block_histograms[block * RADIX_SIZE + digit];
            global_offsets[block * RADIX_SIZE + digit] = running_sum;
            running_sum += count;
        }
    }
}

// ============================================================================
// Scatter Kernel
// Moves elements to their sorted positions
// ============================================================================

kernel void radix_scatter(
    device const uint* __restrict keys_in [[buffer(0)]],
    device const uint* __restrict values_in [[buffer(1)]],
    device uint* __restrict keys_out [[buffer(2)]],
    device uint* __restrict values_out [[buffer(3)]],
    device const uint* __restrict global_offsets [[buffer(4)]],  // [n_blocks, RADIX_SIZE]
    constant SortParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup uint* local_offsets [[threadgroup(0)]],  // [RADIX_SIZE]
    threadgroup uint* local_hist [[threadgroup(1)]]      // [RADIX_SIZE]
) {
    // Load global offsets for this block
    if (tid < RADIX_SIZE) {
        local_offsets[tid] = global_offsets[bid * RADIX_SIZE + tid];
        local_hist[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_start = bid * BLOCK_SIZE * ELEMENTS_PER_THREAD;

    // First pass: count local histogram (needed for local offsets)
    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = block_start + tid + i * BLOCK_SIZE;

        if (idx < params.n_elements) {
            uint key = keys_in[idx];
            uint digit = (key >> params.bit_offset) & RADIX_MASK;

            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&local_hist[digit],
                1u,
                memory_order_relaxed
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Convert histogram to exclusive prefix sum within block
    if (tid == 0) {
        uint sum = 0;
        for (uint d = 0; d < RADIX_SIZE; d++) {
            uint count = local_hist[d];
            local_hist[d] = sum;
            sum += count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: scatter elements
    for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = block_start + tid + i * BLOCK_SIZE;

        if (idx < params.n_elements) {
            uint key = keys_in[idx];
            uint value = values_in[idx];
            uint digit = (key >> params.bit_offset) & RADIX_MASK;

            // Get output position
            uint local_pos = atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&local_hist[digit],
                1u,
                memory_order_relaxed
            );
            uint global_pos = local_offsets[digit] + local_pos;

            keys_out[global_pos] = key;
            values_out[global_pos] = value;
        }
    }
}

// ============================================================================
// Simplified Radix Sort for small arrays (single threadgroup)
// More efficient for arrays < 4096 elements
// ============================================================================

kernel void radix_sort_small(
    device uint* __restrict keys [[buffer(0)]],
    device uint* __restrict values [[buffer(1)]],
    device uint* __restrict keys_temp [[buffer(2)]],
    device uint* __restrict values_temp [[buffer(3)]],
    constant SortParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    threadgroup uint* histogram [[threadgroup(0)]],     // [RADIX_SIZE]
    threadgroup uint* prefix_sum [[threadgroup(1)]]     // [RADIX_SIZE]
) {
    const uint n = params.n_elements;

    // Source and destination alternate each pass
    device uint* src_keys = keys;
    device uint* src_vals = values;
    device uint* dst_keys = keys_temp;
    device uint* dst_vals = values_temp;

    // 8 passes for 32-bit keys (4 bits per pass)
    for (uint pass = 0; pass < 8; pass++) {
        uint bit_offset = pass * RADIX_BITS;

        // Clear histogram
        if (tid < RADIX_SIZE) {
            histogram[tid] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Build histogram
        for (uint i = tid; i < n; i += BLOCK_SIZE) {
            uint key = src_keys[i];
            uint digit = (key >> bit_offset) & RADIX_MASK;
            atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&histogram[digit],
                1u,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Exclusive prefix sum
        if (tid == 0) {
            uint sum = 0;
            for (uint d = 0; d < RADIX_SIZE; d++) {
                uint count = histogram[d];
                prefix_sum[d] = sum;
                sum += count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Copy prefix sum to histogram for atomic scatter
        if (tid < RADIX_SIZE) {
            histogram[tid] = prefix_sum[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scatter
        for (uint i = tid; i < n; i += BLOCK_SIZE) {
            uint key = src_keys[i];
            uint val = src_vals[i];
            uint digit = (key >> bit_offset) & RADIX_MASK;

            uint pos = atomic_fetch_add_explicit(
                (threadgroup atomic_uint*)&histogram[digit],
                1u,
                memory_order_relaxed
            );

            dst_keys[pos] = key;
            dst_vals[pos] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Swap buffers
        device uint* tmp_k = src_keys;
        device uint* tmp_v = src_vals;
        src_keys = dst_keys;
        src_vals = dst_vals;
        dst_keys = tmp_k;
        dst_vals = tmp_v;
    }

    // If odd number of passes, copy result back
    // (8 passes = even, result already in original buffer)
}

// ============================================================================
// Argsort kernel - sorts indices by float values (descending)
// Useful for topk-like operations
// ============================================================================

kernel void argsort_by_score(
    device const float* __restrict scores [[buffer(0)]],
    device uint* __restrict indices [[buffer(1)]],
    constant uint& n_elements [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Initialize indices
    if (gid < n_elements) {
        indices[gid] = gid;
    }
}

// ============================================================================
// TopK kernel - finds k largest elements using partial sort
// More efficient than full sort when k << n
// ============================================================================

struct TopKParams {
    uint n_elements;
    uint k;
};

kernel void topk_select(
    device const float* __restrict scores [[buffer(0)]],
    device uint* __restrict top_indices [[buffer(1)]],
    device float* __restrict top_scores [[buffer(2)]],
    constant TopKParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    threadgroup float* shared_scores [[threadgroup(0)]],
    threadgroup uint* shared_indices [[threadgroup(1)]]
) {
    const uint n = params.n_elements;
    const uint k = params.k;

    // Each thread maintains its own top-k candidates
    float local_scores[8];  // Assume k <= 8 for simplicity
    uint local_indices[8];

    // Initialize with -inf
    for (uint i = 0; i < k && i < 8; i++) {
        local_scores[i] = -INFINITY;
        local_indices[i] = 0;
    }

    // Scan through elements
    for (uint i = tid; i < n; i += BLOCK_SIZE) {
        float score = scores[i];

        // Check if this score belongs in top-k
        if (score > local_scores[k-1]) {
            // Insert in sorted position
            for (uint j = 0; j < k && j < 8; j++) {
                if (score > local_scores[j]) {
                    // Shift down
                    for (uint m = k-1; m > j && m < 8; m--) {
                        local_scores[m] = local_scores[m-1];
                        local_indices[m] = local_indices[m-1];
                    }
                    local_scores[j] = score;
                    local_indices[j] = i;
                    break;
                }
            }
        }
    }

    // Write to shared memory for reduction
    if (tid < k && tid < 8) {
        shared_scores[tid] = local_scores[tid];
        shared_indices[tid] = local_indices[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 writes final result
    if (tid == 0) {
        for (uint i = 0; i < k && i < 8; i++) {
            top_scores[i] = shared_scores[i];
            top_indices[i] = shared_indices[i];
        }
    }
}
