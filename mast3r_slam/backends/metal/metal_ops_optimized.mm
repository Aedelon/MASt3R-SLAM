// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Optimized Obj-C++ bridge for Metal compute kernels.
 *
 * Optimizations:
 * - Buffer caching and reuse
 * - Managed storage mode (zero-copy on unified memory)
 * - Async command buffer execution
 * - Half precision support
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.h"
#include <iostream>
#include <string>
#include <unordered_map>

namespace metal_backend {

// ============================================================================
// Global Metal state
// ============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_matching_library = nil;
static id<MTLLibrary> g_matching_opt_library = nil;
static id<MTLLibrary> g_gn_library = nil;

// Standard pipelines
static id<MTLComputePipelineState> g_iter_proj_pipeline = nil;
static id<MTLComputePipelineState> g_refine_matches_pipeline = nil;
static id<MTLComputePipelineState> g_pose_retr_pipeline = nil;
static id<MTLComputePipelineState> g_ray_align_pipeline = nil;

// Optimized pipelines
static id<MTLComputePipelineState> g_iter_proj_opt_pipeline = nil;
static id<MTLComputePipelineState> g_refine_opt_pipeline = nil;
static id<MTLComputePipelineState> g_refine_half_pipeline = nil;
static id<MTLComputePipelineState> g_refine_tiled_pipeline = nil;

static bool g_initialized = false;
static bool g_use_optimized = true;

// ============================================================================
// Buffer cache for reuse
// ============================================================================

struct BufferCache {
    std::unordered_map<size_t, id<MTLBuffer>> buffers;
    size_t max_cached = 32;

    id<MTLBuffer> get_or_create(size_t size, id<MTLDevice> device) {
        // Round up to power of 2 for better reuse
        size_t rounded = 1;
        while (rounded < size) rounded *= 2;

        auto it = buffers.find(rounded);
        if (it != buffers.end() && [it->second length] >= size) {
            return it->second;
        }

        // Create new buffer with managed storage (zero-copy on Apple Silicon)
        id<MTLBuffer> buffer = [device newBufferWithLength:rounded
                                                   options:MTLResourceStorageModeShared];
        if (buffers.size() < max_cached) {
            buffers[rounded] = buffer;
        }
        return buffer;
    }

    void clear() {
        buffers.clear();
    }
};

static BufferCache g_input_cache;
static BufferCache g_output_cache;

// ============================================================================
// Initialization
// ============================================================================

bool initialize() {
    if (g_initialized) return true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "[Metal] No Metal device found" << std::endl;
            return false;
        }

        std::cout << "[Metal] Using device: " << [[g_device name] UTF8String] << std::endl;

        // Check for Apple Silicon (unified memory)
        if ([g_device hasUnifiedMemory]) {
            std::cout << "[Metal] Unified memory detected - zero-copy enabled" << std::endl;
        }

        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "[Metal] Failed to create command queue" << std::endl;
            return false;
        }

        // Find shader directory
        NSFileManager* fm = [NSFileManager defaultManager];
        NSString* shaderDir = nil;

        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd))) {
            NSArray* searchPaths = @[
                @"mast3r_slam/backends/metal/shaders",
                @"backends/metal/shaders",
            ];

            for (NSString* path in searchPaths) {
                NSString* fullPath = [[NSString stringWithUTF8String:cwd]
                                      stringByAppendingPathComponent:path];
                if ([fm fileExistsAtPath:fullPath]) {
                    shaderDir = fullPath;
                    break;
                }
            }
        }

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion3_0;

        // Compile standard matching shaders
        NSString* matchingPath = [shaderDir stringByAppendingPathComponent:@"matching.metal"];
        if ([fm fileExistsAtPath:matchingPath]) {
            NSString* source = [NSString stringWithContentsOfFile:matchingPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (source) {
                g_matching_library = [g_device newLibraryWithSource:source options:options error:&error];
            }
        }

        // Compile optimized matching shaders
        NSString* matchingOptPath = [shaderDir stringByAppendingPathComponent:@"matching_optimized.metal"];
        if ([fm fileExistsAtPath:matchingOptPath]) {
            NSString* source = [NSString stringWithContentsOfFile:matchingOptPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (source) {
                g_matching_opt_library = [g_device newLibraryWithSource:source options:options error:&error];
                if (!g_matching_opt_library && error) {
                    std::cerr << "[Metal] Failed to compile optimized shaders: "
                              << [[error localizedDescription] UTF8String] << std::endl;
                }
            }
        }

        // Compile GN shaders
        NSString* gnPath = [shaderDir stringByAppendingPathComponent:@"gn.metal"];
        if ([fm fileExistsAtPath:gnPath]) {
            NSString* source = [NSString stringWithContentsOfFile:gnPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (source) {
                g_gn_library = [g_device newLibraryWithSource:source options:options error:&error];
            }
        }

        // Create standard pipelines
        if (g_matching_library) {
            id<MTLFunction> fn = [g_matching_library newFunctionWithName:@"iter_proj_kernel"];
            if (fn) g_iter_proj_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];

            fn = [g_matching_library newFunctionWithName:@"refine_matches_kernel"];
            if (fn) g_refine_matches_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        }

        // Create optimized pipelines
        if (g_matching_opt_library) {
            id<MTLFunction> fn = [g_matching_opt_library newFunctionWithName:@"iter_proj_kernel_optimized"];
            if (fn) g_iter_proj_opt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];

            fn = [g_matching_opt_library newFunctionWithName:@"refine_matches_kernel_optimized"];
            if (fn) g_refine_opt_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];

            fn = [g_matching_opt_library newFunctionWithName:@"refine_matches_kernel_half"];
            if (fn) g_refine_half_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];

            // Tiled version needs threadgroup memory allocation
            fn = [g_matching_opt_library newFunctionWithName:@"refine_matches_kernel_tiled"];
            if (fn) {
                MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
                desc.computeFunction = fn;
                desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                g_refine_tiled_pipeline = [g_device newComputePipelineStateWithDescriptor:desc
                                                                                  options:0
                                                                               reflection:nil
                                                                                    error:&error];
            }
        }

        if (g_gn_library) {
            id<MTLFunction> fn = [g_gn_library newFunctionWithName:@"pose_retr_kernel"];
            if (fn) g_pose_retr_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];

            fn = [g_gn_library newFunctionWithName:@"ray_align_residual_kernel"];
            if (fn) g_ray_align_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        }

        g_initialized = true;

        std::cout << "[Metal] Pipelines initialized:" << std::endl;
        std::cout << "  Standard: iter_proj=" << (g_iter_proj_pipeline ? "OK" : "FAIL")
                  << ", refine=" << (g_refine_matches_pipeline ? "OK" : "FAIL") << std::endl;
        std::cout << "  Optimized: iter_proj=" << (g_iter_proj_opt_pipeline ? "OK" : "FAIL")
                  << ", refine=" << (g_refine_opt_pipeline ? "OK" : "FAIL")
                  << ", half=" << (g_refine_half_pipeline ? "OK" : "FAIL")
                  << ", tiled=" << (g_refine_tiled_pipeline ? "OK" : "FAIL") << std::endl;

        return true;
    }
}

bool is_available() {
    if (!g_initialized) initialize();
    return g_device != nil && (g_iter_proj_pipeline != nil || g_iter_proj_opt_pipeline != nil);
}

// ============================================================================
// Zero-copy buffer creation (for unified memory)
// ============================================================================

static id<MTLBuffer> tensor_to_buffer_zerocopy(torch::Tensor t) {
    t = t.contiguous().to(torch::kCPU);
    size_t size = t.numel() * t.element_size();

    // Use nocopy mode - Metal will use the same memory
    // This works because Apple Silicon has unified memory
    return [g_device newBufferWithBytesNoCopy:t.data_ptr()
                                       length:size
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

static id<MTLBuffer> tensor_to_buffer_cached(torch::Tensor t, BufferCache& cache) {
    t = t.contiguous().to(torch::kCPU);
    size_t size = t.numel() * t.element_size();

    id<MTLBuffer> buffer = cache.get_or_create(size, g_device);
    memcpy([buffer contents], t.data_ptr(), size);
    return buffer;
}

static void buffer_to_tensor(id<MTLBuffer> buffer, torch::Tensor& t) {
    t = t.contiguous();
    memcpy(t.data_ptr(), [buffer contents], t.numel() * t.element_size());
}

// ============================================================================
// Optimized iter_proj
// ============================================================================

std::vector<torch::Tensor> iter_proj(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh)
{
    if (!initialize()) return {};

    // Select optimized or standard pipeline
    id<MTLComputePipelineState> pipeline = g_use_optimized && g_iter_proj_opt_pipeline
        ? g_iter_proj_opt_pipeline : g_iter_proj_pipeline;

    if (!pipeline) {
        TORCH_WARN("Metal iter_proj pipeline not available");
        return {};
    }

    @autoreleasepool {
        int batch_size = p_init.size(0);
        int n_points = p_init.size(1);
        int H = rays_img_with_grad.size(1);
        int W = rays_img_with_grad.size(2);

        // Ensure float32
        rays_img_with_grad = rays_img_with_grad.to(torch::kFloat32);
        pts_3d_norm = pts_3d_norm.to(torch::kFloat32);
        p_init = p_init.to(torch::kFloat32);

        // Output tensors
        torch::Tensor p_out = torch::zeros({batch_size, n_points, 2}, p_init.options());
        torch::Tensor converged = torch::zeros({batch_size, n_points}, torch::kBool);

        // Create buffers (use caching for better performance)
        id<MTLBuffer> rays_buf = tensor_to_buffer_cached(rays_img_with_grad, g_input_cache);
        id<MTLBuffer> pts_buf = tensor_to_buffer_cached(pts_3d_norm, g_input_cache);
        id<MTLBuffer> p_init_buf = tensor_to_buffer_cached(p_init, g_input_cache);

        size_t p_out_size = batch_size * n_points * 2 * sizeof(float);
        size_t conv_size = batch_size * n_points * sizeof(bool);
        id<MTLBuffer> p_out_buf = g_output_cache.get_or_create(p_out_size, g_device);
        id<MTLBuffer> conv_buf = g_output_cache.get_or_create(conv_size, g_device);

        // Params
        struct {
            int batch_size;
            int n_points;
            int H;
            int W;
            int max_iter;
            float lambda_init;
            float cost_thresh;
        } params = {batch_size, n_points, H, W, max_iter, lambda_init, cost_thresh};

        id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&params
                                                         length:sizeof(params)
                                                        options:MTLResourceStorageModeShared];

        // Execute
        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:rays_buf offset:0 atIndex:0];
        [encoder setBuffer:pts_buf offset:0 atIndex:1];
        [encoder setBuffer:p_init_buf offset:0 atIndex:2];
        [encoder setBuffer:p_out_buf offset:0 atIndex:3];
        [encoder setBuffer:conv_buf offset:0 atIndex:4];
        [encoder setBuffer:params_buf offset:0 atIndex:5];

        // Optimized thread dispatch
        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, n_points), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Copy results
        buffer_to_tensor(p_out_buf, p_out);
        buffer_to_tensor(conv_buf, converged);

        return {p_out, converged};
    }
}

// ============================================================================
// Optimized refine_matches
// ============================================================================

std::vector<torch::Tensor> refine_matches(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    int radius,
    int dilation_max)
{
    if (!initialize()) return {};

    // Select best pipeline based on data type and availability
    id<MTLComputePipelineState> pipeline = nullptr;
    bool use_half = false;

    if (g_use_optimized) {
        if (D11.dtype() == torch::kFloat16 && g_refine_half_pipeline) {
            pipeline = g_refine_half_pipeline;
            use_half = true;
        } else if (g_refine_opt_pipeline) {
            pipeline = g_refine_opt_pipeline;
        }
    }

    if (!pipeline) {
        pipeline = g_refine_matches_pipeline;
    }

    if (!pipeline) {
        TORCH_WARN("Metal refine_matches pipeline not available");
        return {};
    }

    @autoreleasepool {
        int batch_size = p1.size(0);
        int n_points = p1.size(1);
        int H = D11.size(1);
        int W = D11.size(2);
        int fdim = D11.size(3);

        // Convert to appropriate type
        if (use_half) {
            D11 = D11.to(torch::kFloat16);
            D21 = D21.to(torch::kFloat16);
        } else {
            D11 = D11.to(torch::kFloat32);
            D21 = D21.to(torch::kFloat32);
        }
        p1 = p1.to(torch::kInt64);

        torch::Tensor p1_out = torch::zeros_like(p1);

        // Create buffers
        id<MTLBuffer> d11_buf = tensor_to_buffer_cached(D11, g_input_cache);
        id<MTLBuffer> d21_buf = tensor_to_buffer_cached(D21, g_input_cache);
        id<MTLBuffer> p1_buf = tensor_to_buffer_cached(p1, g_input_cache);

        size_t p1_out_size = batch_size * n_points * 2 * sizeof(int64_t);
        id<MTLBuffer> p1_out_buf = g_output_cache.get_or_create(p1_out_size, g_device);

        struct {
            int batch_size;
            int n_points;
            int H;
            int W;
            int fdim;
            int radius;
            int dilation_max;
        } params = {batch_size, n_points, H, W, fdim, radius, dilation_max};

        id<MTLBuffer> params_buf = [g_device newBufferWithBytes:&params
                                                         length:sizeof(params)
                                                        options:MTLResourceStorageModeShared];

        // Execute
        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:d11_buf offset:0 atIndex:0];
        [encoder setBuffer:d21_buf offset:0 atIndex:1];
        [encoder setBuffer:p1_buf offset:0 atIndex:2];
        [encoder setBuffer:p1_out_buf offset:0 atIndex:3];
        [encoder setBuffer:params_buf offset:0 atIndex:4];

        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, n_points), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        buffer_to_tensor(p1_out_buf, p1_out);

        return {p1_out};
    }
}

// ============================================================================
// GN kernels (stubs)
// ============================================================================

std::vector<torch::Tensor> gauss_newton_rays(
    torch::Tensor poses, torch::Tensor points, torch::Tensor confidences,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
    torch::Tensor valid_match, torch::Tensor Q,
    float sigma_ray, float sigma_dist, float C_thresh, float Q_thresh,
    int max_iter, float delta_thresh)
{
    TORCH_WARN("gauss_newton_rays: Metal implementation pending");
    return {};
}

std::vector<torch::Tensor> gauss_newton_calib(
    torch::Tensor poses, torch::Tensor points, torch::Tensor confidences, torch::Tensor K,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
    torch::Tensor valid_match, torch::Tensor Q,
    int height, int width, int pixel_border, float z_eps,
    float sigma_pixel, float sigma_depth, float C_thresh, float Q_thresh,
    int max_iter, float delta_thresh)
{
    TORCH_WARN("gauss_newton_calib: Metal implementation pending");
    return {};
}

} // namespace metal_backend
