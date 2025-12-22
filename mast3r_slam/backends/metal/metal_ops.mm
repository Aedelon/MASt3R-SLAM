// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Obj-C++ bridge for Metal compute kernels.
 * Handles Metal device initialization, shader compilation, and kernel dispatch.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.h"
#include <iostream>
#include <string>

namespace metal_backend {

// Global Metal state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_matching_library = nil;
static id<MTLLibrary> g_gn_library = nil;

// Compute pipelines
static id<MTLComputePipelineState> g_iter_proj_pipeline = nil;
static id<MTLComputePipelineState> g_refine_matches_pipeline = nil;
static id<MTLComputePipelineState> g_pose_retr_pipeline = nil;
static id<MTLComputePipelineState> g_ray_align_pipeline = nil;

static bool g_initialized = false;

// ============================================================================
// Initialization
// ============================================================================

bool initialize() {
    if (g_initialized) return true;

    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "[Metal] No Metal device found" << std::endl;
            return false;
        }

        std::cout << "[Metal] Using device: " << [[g_device name] UTF8String] << std::endl;

        // Create command queue
        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "[Metal] Failed to create command queue" << std::endl;
            return false;
        }

        // Find shader directory
        NSBundle* bundle = [NSBundle mainBundle];
        NSString* shaderDir = nil;

        // Try different paths for shader files
        NSArray* searchPaths = @[
            @"mast3r_slam/backends/metal/shaders",
            @"backends/metal/shaders",
            @"shaders"
        ];

        NSFileManager* fm = [NSFileManager defaultManager];
        for (NSString* path in searchPaths) {
            NSString* fullPath = [[[NSBundle mainBundle] bundlePath]
                                  stringByAppendingPathComponent:path];
            if ([fm fileExistsAtPath:fullPath]) {
                shaderDir = fullPath;
                break;
            }
        }

        // If not found, try relative to current directory
        if (!shaderDir) {
            char cwd[1024];
            if (getcwd(cwd, sizeof(cwd))) {
                for (NSString* path in searchPaths) {
                    NSString* fullPath = [[NSString stringWithUTF8String:cwd]
                                          stringByAppendingPathComponent:path];
                    if ([fm fileExistsAtPath:fullPath]) {
                        shaderDir = fullPath;
                        break;
                    }
                }
            }
        }

        if (!shaderDir) {
            std::cerr << "[Metal] Shader directory not found" << std::endl;
            // Continue anyway - shaders will be compiled at runtime
        }

        NSError* error = nil;

        // Compile matching shaders
        NSString* matchingPath = [shaderDir stringByAppendingPathComponent:@"matching.metal"];
        if ([fm fileExistsAtPath:matchingPath]) {
            NSString* source = [NSString stringWithContentsOfFile:matchingPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (source) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.fastMathEnabled = YES;
                g_matching_library = [g_device newLibraryWithSource:source options:options error:&error];
                if (!g_matching_library) {
                    std::cerr << "[Metal] Failed to compile matching shaders: "
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
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.fastMathEnabled = YES;
                g_gn_library = [g_device newLibraryWithSource:source options:options error:&error];
                if (!g_gn_library) {
                    std::cerr << "[Metal] Failed to compile GN shaders: "
                              << [[error localizedDescription] UTF8String] << std::endl;
                }
            }
        }

        // Create compute pipelines
        if (g_matching_library) {
            id<MTLFunction> iter_proj_fn = [g_matching_library newFunctionWithName:@"iter_proj_kernel"];
            if (iter_proj_fn) {
                g_iter_proj_pipeline = [g_device newComputePipelineStateWithFunction:iter_proj_fn error:&error];
            }

            id<MTLFunction> refine_fn = [g_matching_library newFunctionWithName:@"refine_matches_kernel"];
            if (refine_fn) {
                g_refine_matches_pipeline = [g_device newComputePipelineStateWithFunction:refine_fn error:&error];
            }
        }

        if (g_gn_library) {
            id<MTLFunction> pose_retr_fn = [g_gn_library newFunctionWithName:@"pose_retr_kernel"];
            if (pose_retr_fn) {
                g_pose_retr_pipeline = [g_device newComputePipelineStateWithFunction:pose_retr_fn error:&error];
            }

            id<MTLFunction> ray_align_fn = [g_gn_library newFunctionWithName:@"ray_align_residual_kernel"];
            if (ray_align_fn) {
                g_ray_align_pipeline = [g_device newComputePipelineStateWithFunction:ray_align_fn error:&error];
            }
        }

        g_initialized = true;
        std::cout << "[Metal] Initialization complete" << std::endl;
        std::cout << "[Metal] Pipelines: iter_proj=" << (g_iter_proj_pipeline ? "OK" : "FAIL")
                  << ", refine=" << (g_refine_matches_pipeline ? "OK" : "FAIL")
                  << ", pose_retr=" << (g_pose_retr_pipeline ? "OK" : "FAIL")
                  << ", ray_align=" << (g_ray_align_pipeline ? "OK" : "FAIL") << std::endl;

        return true;
    }
}

bool is_available() {
    if (!g_initialized) {
        initialize();
    }
    return g_device != nil && g_iter_proj_pipeline != nil;
}

// ============================================================================
// Helper: Create Metal buffer from tensor
// ============================================================================

static id<MTLBuffer> tensor_to_buffer(torch::Tensor t, bool copy_data = true) {
    t = t.contiguous().to(torch::kCPU);
    size_t size = t.numel() * t.element_size();

    if (copy_data) {
        return [g_device newBufferWithBytes:t.data_ptr()
                                     length:size
                                    options:MTLResourceStorageModeShared];
    } else {
        return [g_device newBufferWithLength:size
                                     options:MTLResourceStorageModeShared];
    }
}

static void buffer_to_tensor(id<MTLBuffer> buffer, torch::Tensor& t) {
    t = t.contiguous();
    memcpy(t.data_ptr(), [buffer contents], t.numel() * t.element_size());
}

// ============================================================================
// iter_proj implementation
// ============================================================================

std::vector<torch::Tensor> iter_proj(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh)
{
    if (!initialize() || !g_iter_proj_pipeline) {
        TORCH_WARN("Metal iter_proj not available, returning empty");
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

        // Create output tensors
        torch::Tensor p_out = torch::zeros({batch_size, n_points, 2}, p_init.options());
        torch::Tensor converged = torch::zeros({batch_size, n_points}, torch::kBool);

        // Create Metal buffers
        id<MTLBuffer> rays_buf = tensor_to_buffer(rays_img_with_grad);
        id<MTLBuffer> pts_buf = tensor_to_buffer(pts_3d_norm);
        id<MTLBuffer> p_init_buf = tensor_to_buffer(p_init);
        id<MTLBuffer> p_out_buf = tensor_to_buffer(p_out, false);
        id<MTLBuffer> conv_buf = tensor_to_buffer(converged, false);

        // Create params buffer
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

        // Create command buffer and encoder
        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_iter_proj_pipeline];
        [encoder setBuffer:rays_buf offset:0 atIndex:0];
        [encoder setBuffer:pts_buf offset:0 atIndex:1];
        [encoder setBuffer:p_init_buf offset:0 atIndex:2];
        [encoder setBuffer:p_out_buf offset:0 atIndex:3];
        [encoder setBuffer:conv_buf offset:0 atIndex:4];
        [encoder setBuffer:params_buf offset:0 atIndex:5];

        // Dispatch
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(
            MIN(64, g_iter_proj_pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Copy results back
        buffer_to_tensor(p_out_buf, p_out);
        buffer_to_tensor(conv_buf, converged);

        return {p_out, converged};
    }
}

// ============================================================================
// refine_matches implementation
// ============================================================================

std::vector<torch::Tensor> refine_matches(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    int radius,
    int dilation_max)
{
    if (!initialize() || !g_refine_matches_pipeline) {
        TORCH_WARN("Metal refine_matches not available, returning empty");
        return {};
    }

    @autoreleasepool {
        int batch_size = p1.size(0);
        int n_points = p1.size(1);
        int H = D11.size(1);
        int W = D11.size(2);
        int fdim = D11.size(3);

        // Ensure correct types
        D11 = D11.to(torch::kFloat32);
        D21 = D21.to(torch::kFloat32);
        p1 = p1.to(torch::kInt64);

        // Create output tensor
        torch::Tensor p1_out = torch::zeros_like(p1);

        // Create Metal buffers
        id<MTLBuffer> d11_buf = tensor_to_buffer(D11);
        id<MTLBuffer> d21_buf = tensor_to_buffer(D21);
        id<MTLBuffer> p1_buf = tensor_to_buffer(p1);
        id<MTLBuffer> p1_out_buf = tensor_to_buffer(p1_out, false);

        // Create params buffer
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

        // Create command buffer and encoder
        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:g_refine_matches_pipeline];
        [encoder setBuffer:d11_buf offset:0 atIndex:0];
        [encoder setBuffer:d21_buf offset:0 atIndex:1];
        [encoder setBuffer:p1_buf offset:0 atIndex:2];
        [encoder setBuffer:p1_out_buf offset:0 atIndex:3];
        [encoder setBuffer:params_buf offset:0 atIndex:4];

        // Dispatch
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(
            MIN(64, g_refine_matches_pipeline.maxTotalThreadsPerThreadgroup), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Copy results back
        buffer_to_tensor(p1_out_buf, p1_out);

        return {p1_out};
    }
}

// ============================================================================
// Gauss-Newton kernels (stubs for now)
// ============================================================================

std::vector<torch::Tensor> gauss_newton_rays(
    torch::Tensor poses,
    torch::Tensor points,
    torch::Tensor confidences,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor idx_ii2jj,
    torch::Tensor valid_match,
    torch::Tensor Q,
    float sigma_ray,
    float sigma_dist,
    float C_thresh,
    float Q_thresh,
    int max_iter,
    float delta_thresh)
{
    // TODO: Full Metal implementation
    TORCH_WARN("gauss_newton_rays: Metal implementation pending");
    return {};
}

std::vector<torch::Tensor> gauss_newton_calib(
    torch::Tensor poses,
    torch::Tensor points,
    torch::Tensor confidences,
    torch::Tensor K,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor idx_ii2jj,
    torch::Tensor valid_match,
    torch::Tensor Q,
    int height,
    int width,
    int pixel_border,
    float z_eps,
    float sigma_pixel,
    float sigma_depth,
    float C_thresh,
    float Q_thresh,
    int max_iter,
    float delta_thresh)
{
    // TODO: Full Metal implementation
    TORCH_WARN("gauss_newton_calib: Metal implementation pending");
    return {};
}

} // namespace metal_backend
