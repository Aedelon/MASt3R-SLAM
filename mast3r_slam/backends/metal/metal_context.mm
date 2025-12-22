// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.

#import "metal_context.h"
#include <iostream>
#include <unistd.h>

namespace metal_backend {

// ============================================================================
// BufferPool implementation
// ============================================================================

size_t BufferPool::round_up_power_of_2(size_t size) {
    size_t rounded = 1;
    while (rounded < size) rounded *= 2;
    return rounded;
}

id<MTLBuffer> BufferPool::acquire(size_t size, id<MTLDevice> device) {
    size_t rounded = round_up_power_of_2(size);

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_buffers_.find(rounded);
    if (it != free_buffers_.end() && !it->second.empty()) {
        id<MTLBuffer> buffer = it->second.back();
        it->second.pop_back();
        return buffer;
    }

    // Create new buffer with shared storage (zero-copy on Apple Silicon)
    return [device newBufferWithLength:rounded
                               options:MTLResourceStorageModeShared];
}

void BufferPool::release(id<MTLBuffer> buffer) {
    if (!buffer) return;

    size_t size = [buffer length];
    std::lock_guard<std::mutex> lock(mutex_);

    auto& vec = free_buffers_[size];
    if (vec.size() < max_cached_per_size_) {
        vec.push_back(buffer);
    }
    // Otherwise let ARC release it
}

void BufferPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers_.clear();
}

size_t BufferPool::total_bytes() const {
    size_t total = 0;
    for (const auto& pair : free_buffers_) {
        total += pair.first * pair.second.size();
    }
    return total;
}

// ============================================================================
// MetalContext implementation
// ============================================================================

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

MetalContext::~MetalContext() {
    input_pool_.clear();
    output_pool_.clear();
    pipelines_.clear();
}

bool MetalContext::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    @autoreleasepool {
        // Get default Metal device
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "[MetalContext] No Metal device found" << std::endl;
            return false;
        }

        device_name_ = [[device_ name] UTF8String];
        has_unified_memory_ = [device_ hasUnifiedMemory];

        std::cout << "[MetalContext] Device: " << device_name_ << std::endl;
        if (has_unified_memory_) {
            std::cout << "[MetalContext] Unified memory: enabled (zero-copy)" << std::endl;
        }

        // Create command queue
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            std::cerr << "[MetalContext] Failed to create command queue" << std::endl;
            return false;
        }

        // Compile shaders and create pipelines
        if (!compile_shaders()) {
            std::cerr << "[MetalContext] Failed to compile shaders" << std::endl;
            return false;
        }

        if (!create_pipelines()) {
            std::cerr << "[MetalContext] Failed to create pipelines" << std::endl;
            return false;
        }

        initialized_ = true;

        // Log pipeline status
        std::cout << "[MetalContext] Pipelines:" << std::endl;
        for (const auto& pair : pipelines_) {
            std::cout << "  " << pair.first << ": "
                      << (pair.second ? "OK" : "FAIL") << std::endl;
        }

        return true;
    }
}

bool MetalContext::is_available() const {
    return initialized_ && device_ != nil && !pipelines_.empty();
}

NSString* MetalContext::find_shader_directory() {
    NSFileManager* fm = [NSFileManager defaultManager];

    char cwd[1024];
    if (!getcwd(cwd, sizeof(cwd))) {
        return nil;
    }

    NSArray* searchPaths = @[
        @"mast3r_slam/backends/metal/shaders",
        @"backends/metal/shaders",
    ];

    for (NSString* path in searchPaths) {
        NSString* fullPath = [[NSString stringWithUTF8String:cwd]
                              stringByAppendingPathComponent:path];
        if ([fm fileExistsAtPath:fullPath]) {
            return fullPath;
        }
    }

    return nil;
}

bool MetalContext::compile_shaders() {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* shaderDir = find_shader_directory();

    if (!shaderDir) {
        std::cerr << "[MetalContext] Shader directory not found" << std::endl;
        return false;
    }

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    options.languageVersion = MTLLanguageVersion3_0;

    // Compile matching.metal
    NSString* matchingPath = [shaderDir stringByAppendingPathComponent:@"matching.metal"];
    if ([fm fileExistsAtPath:matchingPath]) {
        NSString* source = [NSString stringWithContentsOfFile:matchingPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            matching_library_ = [device_ newLibraryWithSource:source
                                                      options:options
                                                        error:&error];
            if (!matching_library_ && error) {
                std::cerr << "[MetalContext] matching.metal: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }
    }

    // Compile gn.metal
    NSString* gnPath = [shaderDir stringByAppendingPathComponent:@"gn.metal"];
    if ([fm fileExistsAtPath:gnPath]) {
        NSString* source = [NSString stringWithContentsOfFile:gnPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            gn_library_ = [device_ newLibraryWithSource:source
                                                options:options
                                                  error:&error];
            if (!gn_library_ && error) {
                std::cerr << "[MetalContext] gn.metal: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }
    }

    // Compile sort.metal
    NSString* sortPath = [shaderDir stringByAppendingPathComponent:@"sort.metal"];
    if ([fm fileExistsAtPath:sortPath]) {
        NSString* source = [NSString stringWithContentsOfFile:sortPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            sort_library_ = [device_ newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
            if (!sort_library_ && error) {
                std::cerr << "[MetalContext] sort.metal: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }
    }

    return matching_library_ != nil || gn_library_ != nil || sort_library_ != nil;
}

bool MetalContext::create_pipelines() {
    NSError* error = nil;

    if (matching_library_) {
        // iter_proj
        id<MTLFunction> fn = [matching_library_ newFunctionWithName:@"iter_proj_kernel_optimized"];
        if (fn) {
            pipelines_[pipelines::ITER_PROJ] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // refine_matches
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_optimized"];
        if (fn) {
            pipelines_[pipelines::REFINE_MATCHES] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // refine_from_float (accepts float positions from iter_proj)
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_from_float"];
        if (fn) {
            pipelines_[pipelines::REFINE_FROM_FLOAT] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // refine_half (FP16)
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_half"];
        if (fn) {
            pipelines_[pipelines::REFINE_HALF] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // refine_tiled (threadgroup memory)
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_tiled"];
        if (fn) {
            MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
            desc.computeFunction = fn;
            desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            pipelines_[pipelines::REFINE_TILED] =
                [device_ newComputePipelineStateWithDescriptor:desc
                                                       options:0
                                                    reflection:nil
                                                         error:&error];
        }

        // refine_simd (SIMD-group parallel search)
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_simd"];
        if (fn) {
            MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
            desc.computeFunction = fn;
            desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            pipelines_[pipelines::REFINE_SIMD] =
                [device_ newComputePipelineStateWithDescriptor:desc
                                                       options:0
                                                    reflection:nil
                                                         error:&error];
        }

        // refine_cooperative (cooperative loading)
        fn = [matching_library_ newFunctionWithName:@"refine_matches_kernel_cooperative"];
        if (fn) {
            MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
            desc.computeFunction = fn;
            desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
            pipelines_[pipelines::REFINE_COOPERATIVE] =
                [device_ newComputePipelineStateWithDescriptor:desc
                                                       options:0
                                                    reflection:nil
                                                         error:&error];
        }
    }

    if (gn_library_) {
        // pose_retr
        id<MTLFunction> fn = [gn_library_ newFunctionWithName:@"pose_retr_kernel"];
        if (fn) {
            pipelines_[pipelines::POSE_RETR] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // ray_align
        fn = [gn_library_ newFunctionWithName:@"ray_align_residual_kernel"];
        if (fn) {
            pipelines_[pipelines::RAY_ALIGN] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }
    }

    if (sort_library_) {
        // radix_histogram
        id<MTLFunction> fn = [sort_library_ newFunctionWithName:@"radix_histogram"];
        if (fn) {
            pipelines_[pipelines::RADIX_HISTOGRAM] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // radix_prefix_sum
        fn = [sort_library_ newFunctionWithName:@"radix_prefix_sum"];
        if (fn) {
            pipelines_[pipelines::RADIX_PREFIX_SUM] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // radix_scatter
        fn = [sort_library_ newFunctionWithName:@"radix_scatter"];
        if (fn) {
            pipelines_[pipelines::RADIX_SCATTER] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // radix_sort_small
        fn = [sort_library_ newFunctionWithName:@"radix_sort_small"];
        if (fn) {
            pipelines_[pipelines::RADIX_SORT_SMALL] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }

        // topk_select
        fn = [sort_library_ newFunctionWithName:@"topk_select"];
        if (fn) {
            pipelines_[pipelines::TOPK_SELECT] =
                [device_ newComputePipelineStateWithFunction:fn error:&error];
        }
    }

    return !pipelines_.empty();
}

id<MTLComputePipelineState> MetalContext::get_pipeline(const std::string& name) const {
    auto it = pipelines_.find(name);
    return it != pipelines_.end() ? it->second : nil;
}

bool MetalContext::has_pipeline(const std::string& name) const {
    auto it = pipelines_.find(name);
    return it != pipelines_.end() && it->second != nil;
}

id<MTLCommandBuffer> MetalContext::create_command_buffer() {
    command_buffers_created_++;
    return [command_queue_ commandBuffer];
}

id<MTLBuffer> MetalContext::create_buffer(size_t size, MTLResourceOptions options) {
    return [device_ newBufferWithLength:size options:options];
}

id<MTLBuffer> MetalContext::create_buffer_with_data(const void* data, size_t size) {
    return [device_ newBufferWithBytes:data
                                length:size
                               options:MTLResourceStorageModeShared];
}

id<MTLBuffer> MetalContext::tensor_to_buffer(const void* data, size_t size, BufferPool& pool) {
    id<MTLBuffer> buffer = pool.acquire(size, device_);
    memcpy([buffer contents], data, size);
    return buffer;
}

// ============================================================================
// MPS Zero-Copy Support
// ============================================================================

MetalBuffer MetalContext::prepare_input(torch::Tensor& t, BufferPool& pool) {
    t = t.contiguous();

    if (is_mps_tensor(t)) {
        // Zero-copy: use MPS buffer directly
        // Use commit() instead of synchronize() - non-blocking, just submits commands
        // Our command buffer will naturally wait for MPS work due to buffer dependency
        if (torch::mps::is_available()) {
            torch::mps::commit();  // Submit pending MPS work (non-blocking)
        }

        id<MTLBuffer> buffer = get_mps_buffer(t);
        size_t offset = t.storage_offset() * t.element_size();
        return MetalBuffer(buffer, true, offset);
    } else {
        // Copy to pooled buffer
        t = t.to(torch::kCPU);
        size_t size = t.numel() * t.element_size();
        id<MTLBuffer> buffer = pool.acquire(size, device_);
        memcpy([buffer contents], t.data_ptr(), size);
        return MetalBuffer(buffer, false, 0);
    }
}

std::pair<MetalBuffer, torch::Tensor> MetalContext::prepare_output(
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    bool use_mps)
{
    if (use_mps) {
        // Create MPS tensor and use its buffer directly
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kMPS);
        torch::Tensor t = torch::empty(shape, options);

        // Commit pending MPS allocations (non-blocking)
        if (torch::mps::is_available()) {
            torch::mps::commit();
        }

        id<MTLBuffer> buffer = get_mps_buffer(t);
        return {MetalBuffer(buffer, true, 0), t};
    } else {
        // Create CPU tensor and use pooled buffer
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        torch::Tensor t = torch::empty(shape, options);

        size_t size = t.numel() * t.element_size();
        id<MTLBuffer> buffer = output_pool_.acquire(size, device_);
        return {MetalBuffer(buffer, false, 0), t};
    }
}

void MetalContext::finalize_output(const MetalBuffer& buf, torch::Tensor& t) {
    if (buf.zero_copy) {
        // MPS tensor: data is already in place
        // No sync needed - we already waited on our command buffer with waitUntilCompleted
        // The GPU work is guaranteed complete at this point
    } else {
        // CPU tensor: copy from buffer
        t = t.contiguous();
        memcpy(t.data_ptr(), [buf.buffer contents], t.numel() * t.element_size());
    }
}

void MetalContext::sync_mps_stream() {
    // Synchronize PyTorch MPS stream with our Metal operations
    // This ensures all pending MPS operations are complete
    if (torch::mps::is_available()) {
        torch::mps::synchronize();
    }
}

MetalContext::Stats MetalContext::get_stats() const {
    return {
        input_pool_.cached_count(),
        output_pool_.cached_count(),
        pipelines_.size(),
        command_buffers_created_
    };
}

// ============================================================================
// CommandBatch implementation
// ============================================================================

CommandBatch::CommandBatch(MetalContext& ctx) : ctx_(ctx) {
    cmd_buf_ = ctx.create_command_buffer();
}

CommandBatch::~CommandBatch() {
    if (!committed_ && encoder_count_ > 0) {
        // Auto-commit if not already done
        commit_and_wait();
    }
}

id<MTLComputeCommandEncoder> CommandBatch::add_encoder() {
    if (committed_) {
        std::cerr << "[CommandBatch] Cannot add encoder after commit" << std::endl;
        return nil;
    }
    encoder_count_++;
    return [cmd_buf_ computeCommandEncoder];
}

void CommandBatch::commit_and_wait() {
    if (committed_) return;
    committed_ = true;
    [cmd_buf_ commit];
    [cmd_buf_ waitUntilCompleted];
}

void CommandBatch::commit() {
    if (committed_) return;
    committed_ = true;
    [cmd_buf_ commit];
}

void CommandBatch::wait() {
    if (!committed_) {
        commit();
    }
    [cmd_buf_ waitUntilCompleted];
}

// ============================================================================
// SyncPoint implementation
// ============================================================================

SyncPoint::SyncPoint(MetalContext& ctx) {
    event_ = [ctx.device() newSharedEvent];
    [event_ setSignaledValue:0];
}

void SyncPoint::signal_after(id<MTLCommandBuffer> cmd_buf) {
    value_++;
    [cmd_buf encodeSignalEvent:event_ value:value_];
}

void SyncPoint::wait_before(id<MTLCommandBuffer> cmd_buf) {
    [cmd_buf encodeWaitForEvent:event_ value:value_];
}

} // namespace metal_backend
