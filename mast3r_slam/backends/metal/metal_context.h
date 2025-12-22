// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <torch/torch.h>

namespace metal_backend {

// ============================================================================
// MPS Zero-Copy Utilities
// ============================================================================

/**
 * Check if a tensor is on MPS device.
 */
inline bool is_mps_tensor(const torch::Tensor& t) {
    return t.device().type() == torch::kMPS;
}

/**
 * Get MTLBuffer from MPS tensor storage (zero-copy).
 * Uses bit_cast to reinterpret the storage pointer as MTLBuffer.
 * Only valid for MPS tensors!
 */
inline id<MTLBuffer> get_mps_buffer(const torch::Tensor& t) {
    // MPS tensors store their data in MTLBuffer
    // The storage data pointer IS the MTLBuffer pointer
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

/**
 * Result of preparing a tensor for Metal operations.
 * Contains the buffer and whether it's zero-copy (no cleanup needed).
 */
struct MetalBuffer {
    id<MTLBuffer> buffer;
    bool zero_copy;      // True if buffer points to MPS tensor storage
    size_t offset;       // Offset into buffer (for views)

    MetalBuffer() : buffer(nil), zero_copy(false), offset(0) {}
    MetalBuffer(id<MTLBuffer> b, bool zc, size_t off = 0)
        : buffer(b), zero_copy(zc), offset(off) {}
};

/**
 * Thread-safe buffer pool for Metal buffer reuse.
 * Reduces allocation overhead by caching buffers by size.
 */
class BufferPool {
public:
    id<MTLBuffer> acquire(size_t size, id<MTLDevice> device);
    void release(id<MTLBuffer> buffer);
    void clear();

    size_t cached_count() const { return free_buffers_.size(); }
    size_t total_bytes() const;

private:
    static size_t round_up_power_of_2(size_t size);

    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<id<MTLBuffer>>> free_buffers_;
    size_t max_cached_per_size_ = 8;
};

/**
 * MetalContext - Singleton managing all Metal resources.
 *
 * Provides:
 * - Device and command queue management
 * - Shader compilation and pipeline caching
 * - Buffer pooling for reduced allocation overhead
 * - Thread-safe access to Metal resources
 *
 * Usage:
 *   auto& ctx = MetalContext::instance();
 *   if (ctx.is_available()) {
 *       auto* pipeline = ctx.get_pipeline("iter_proj");
 *       auto* cmd_buf = ctx.create_command_buffer();
 *       // ...
 *   }
 */
class MetalContext {
public:
    // Singleton access
    static MetalContext& instance();

    // Delete copy/move
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) = delete;
    MetalContext& operator=(MetalContext&&) = delete;

    // Initialization
    bool initialize();
    bool is_available() const;
    bool is_initialized() const { return initialized_; }

    // Device info
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> command_queue() const { return command_queue_; }
    bool has_unified_memory() const { return has_unified_memory_; }
    const std::string& device_name() const { return device_name_; }

    // Pipeline access
    id<MTLComputePipelineState> get_pipeline(const std::string& name) const;
    bool has_pipeline(const std::string& name) const;

    // Command buffer creation
    id<MTLCommandBuffer> create_command_buffer();

    // Buffer pool access
    BufferPool& input_pool() { return input_pool_; }
    BufferPool& output_pool() { return output_pool_; }

    // Buffer utilities
    id<MTLBuffer> create_buffer(size_t size, MTLResourceOptions options = MTLResourceStorageModeShared);
    id<MTLBuffer> create_buffer_with_data(const void* data, size_t size);

    // Convenience: tensor to buffer (with caching)
    id<MTLBuffer> tensor_to_buffer(const void* data, size_t size, BufferPool& pool);

    // =========================================================================
    // MPS Zero-Copy Support
    // =========================================================================

    /**
     * Prepare input tensor for Metal kernel.
     * - MPS tensor: zero-copy, returns pointer to MPS storage
     * - CPU tensor: copies to pooled buffer
     * Returns MetalBuffer with buffer and zero_copy flag.
     */
    MetalBuffer prepare_input(torch::Tensor& t, BufferPool& pool);

    /**
     * Prepare output buffer for Metal kernel.
     * - MPS mode: creates MPS tensor, returns its buffer (zero-copy)
     * - CPU mode: returns pooled buffer
     * Also returns the output tensor.
     */
    std::pair<MetalBuffer, torch::Tensor> prepare_output(
        const std::vector<int64_t>& shape,
        torch::ScalarType dtype,
        bool use_mps);

    /**
     * Finalize output after kernel execution.
     * - MPS mode: tensor already has data, nothing to do
     * - CPU mode: copies from buffer to tensor
     */
    void finalize_output(const MetalBuffer& buf, torch::Tensor& t);

    /**
     * Synchronize MPS stream with our Metal command buffer.
     * Call before accessing MPS tensor data.
     */
    void sync_mps_stream();

    // Stats
    struct Stats {
        size_t input_buffers_cached;
        size_t output_buffers_cached;
        size_t pipelines_loaded;
        size_t command_buffers_created;
    };
    Stats get_stats() const;

private:
    MetalContext() = default;
    ~MetalContext();

    bool compile_shaders();
    bool create_pipelines();
    NSString* find_shader_directory();

    // Device
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    std::string device_name_;
    bool has_unified_memory_ = false;

    // Libraries
    id<MTLLibrary> matching_library_ = nil;
    id<MTLLibrary> gn_library_ = nil;
    id<MTLLibrary> sort_library_ = nil;

    // Pipelines (name -> pipeline)
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;

    // Buffer pools
    BufferPool input_pool_;
    BufferPool output_pool_;

    // State
    bool initialized_ = false;
    mutable std::mutex mutex_;

    // Stats
    mutable size_t command_buffers_created_ = 0;
};

// ============================================================================
// CommandBatch - Batches multiple kernel dispatches
// ============================================================================

/**
 * CommandBatch accumulates multiple compute dispatches into a single
 * command buffer, reducing synchronization overhead.
 *
 * Usage:
 *   CommandBatch batch(ctx);
 *   auto enc1 = batch.add_encoder();
 *   // ... encode first kernel ...
 *   [enc1 endEncoding];
 *
 *   auto enc2 = batch.add_encoder();
 *   // ... encode second kernel ...
 *   [enc2 endEncoding];
 *
 *   batch.commit_and_wait();
 */
class CommandBatch {
public:
    explicit CommandBatch(MetalContext& ctx);
    ~CommandBatch();

    // Get a new compute encoder for the next kernel
    // Previous encoder must be ended before calling this
    id<MTLComputeCommandEncoder> add_encoder();

    // Get the underlying command buffer (for advanced use)
    id<MTLCommandBuffer> command_buffer() const { return cmd_buf_; }

    // Commit and wait synchronously
    void commit_and_wait();

    // Commit without waiting (async execution)
    void commit();

    // Wait for async execution to complete
    void wait();

    // Check if batch has been committed
    bool is_committed() const { return committed_; }

    // Get number of encoders added
    size_t encoder_count() const { return encoder_count_; }

private:
    MetalContext& ctx_;
    id<MTLCommandBuffer> cmd_buf_;
    bool committed_ = false;
    size_t encoder_count_ = 0;
};

// ============================================================================
// SyncPoint - Fine-grained synchronization between command buffers
// ============================================================================

/**
 * SyncPoint uses MTLSharedEvent for efficient GPU-side synchronization
 * between command buffers without CPU round-trips.
 *
 * Usage:
 *   SyncPoint sync(ctx);
 *
 *   // First command buffer signals when done
 *   sync.signal_after(cmd_buf1);
 *   [cmd_buf1 commit];
 *
 *   // Second command buffer waits for signal
 *   sync.wait_before(cmd_buf2);
 *   [cmd_buf2 commit];
 */
class SyncPoint {
public:
    explicit SyncPoint(MetalContext& ctx);

    // Signal this sync point after command buffer completes
    void signal_after(id<MTLCommandBuffer> cmd_buf);

    // Wait for this sync point before command buffer starts
    void wait_before(id<MTLCommandBuffer> cmd_buf);

    // Get current value
    uint64_t value() const { return value_; }

private:
    id<MTLSharedEvent> event_;
    uint64_t value_ = 0;
};

// Pipeline names (constants)
namespace pipelines {
    constexpr const char* ITER_PROJ = "iter_proj";
    constexpr const char* REFINE_MATCHES = "refine_matches";
    constexpr const char* REFINE_FROM_FLOAT = "refine_from_float"; // Float input (no CPU sync)
    constexpr const char* REFINE_HALF = "refine_half";
    constexpr const char* REFINE_TILED = "refine_tiled";
    constexpr const char* REFINE_SIMD = "refine_simd";           // SIMD-group parallel search
    constexpr const char* REFINE_COOPERATIVE = "refine_cooperative"; // Cooperative loading
    constexpr const char* POSE_RETR = "pose_retr";
    constexpr const char* RAY_ALIGN = "ray_align";

    // Sorting kernels
    constexpr const char* RADIX_HISTOGRAM = "radix_histogram";
    constexpr const char* RADIX_PREFIX_SUM = "radix_prefix_sum";
    constexpr const char* RADIX_SCATTER = "radix_scatter";
    constexpr const char* RADIX_SORT_SMALL = "radix_sort_small";
    constexpr const char* TOPK_SELECT = "topk_select";
}

} // namespace metal_backend
