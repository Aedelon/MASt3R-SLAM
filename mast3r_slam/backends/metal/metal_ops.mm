// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Metal compute operations using MetalContext singleton.
 *
 * Supports two modes:
 * - MPS tensors: Zero-copy, uses MPS buffer storage directly
 * - CPU tensors: Copies to/from pooled Metal buffers
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ops.h"
#include "metal_context.h"
#include <iostream>

namespace metal_backend {

// ============================================================================
// Public API
// ============================================================================

bool initialize() {
    return MetalContext::instance().initialize();
}

bool is_available() {
    auto& ctx = MetalContext::instance();
    if (!ctx.is_initialized()) {
        ctx.initialize();
    }
    return ctx.is_available() && ctx.has_pipeline(pipelines::ITER_PROJ);
}

// ============================================================================
// iter_proj - with MPS zero-copy support
// ============================================================================

std::vector<torch::Tensor> iter_proj(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* pipeline = ctx.get_pipeline(pipelines::ITER_PROJ);
    if (!pipeline) {
        TORCH_WARN("Metal iter_proj pipeline not available");
        return {};
    }

    @autoreleasepool {
        int batch_size = p_init.size(0);
        int n_points = p_init.size(1);
        int H = rays_img_with_grad.size(1);
        int W = rays_img_with_grad.size(2);

        // Determine if we should use MPS mode (if any input is MPS)
        bool use_mps = is_mps_tensor(rays_img_with_grad) ||
                       is_mps_tensor(pts_3d_norm) ||
                       is_mps_tensor(p_init);

        // Ensure float32
        rays_img_with_grad = rays_img_with_grad.to(torch::kFloat32);
        pts_3d_norm = pts_3d_norm.to(torch::kFloat32);
        p_init = p_init.to(torch::kFloat32);

        // Prepare inputs (zero-copy for MPS, copy for CPU)
        MetalBuffer rays_buf = ctx.prepare_input(rays_img_with_grad, ctx.input_pool());
        MetalBuffer pts_buf = ctx.prepare_input(pts_3d_norm, ctx.input_pool());
        MetalBuffer p_init_buf = ctx.prepare_input(p_init, ctx.input_pool());

        // Prepare outputs
        auto [p_out_buf, p_out] = ctx.prepare_output(
            {batch_size, n_points, 2}, torch::kFloat32, use_mps);
        auto [conv_buf, converged] = ctx.prepare_output(
            {batch_size, n_points}, torch::kBool, use_mps);

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

        id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

        // Execute
        id<MTLCommandBuffer> cmd_buf = ctx.create_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:rays_buf.buffer offset:rays_buf.offset atIndex:0];
        [encoder setBuffer:pts_buf.buffer offset:pts_buf.offset atIndex:1];
        [encoder setBuffer:p_init_buf.buffer offset:p_init_buf.offset atIndex:2];
        [encoder setBuffer:p_out_buf.buffer offset:p_out_buf.offset atIndex:3];
        [encoder setBuffer:conv_buf.buffer offset:conv_buf.offset atIndex:4];
        [encoder setBuffer:params_buf offset:0 atIndex:5];

        // Optimized thread dispatch with aligned threadgroup size
        NSUInteger threadWidth = pipeline.threadExecutionWidth;  // typically 32
        NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;

        // Align to threadExecutionWidth for optimal SIMD utilization
        NSUInteger alignedWidth = ((n_points + threadWidth - 1) / threadWidth) * threadWidth;
        alignedWidth = MIN(alignedWidth, maxThreads);
        alignedWidth = MAX(alignedWidth, threadWidth);

        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(alignedWidth, (NSUInteger)n_points), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Finalize outputs (no-op for MPS, copy for CPU)
        ctx.finalize_output(p_out_buf, p_out);
        ctx.finalize_output(conv_buf, converged);

        return {p_out, converged};
    }
}

// ============================================================================
// refine_matches - with MPS zero-copy support
// ============================================================================

std::vector<torch::Tensor> refine_matches(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    int radius,
    int dilation_max)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    // Select best pipeline based on data type
    id<MTLComputePipelineState> pipeline = nullptr;
    bool use_half = false;

    if (D11.dtype() == torch::kFloat16 && ctx.has_pipeline(pipelines::REFINE_HALF)) {
        pipeline = ctx.get_pipeline(pipelines::REFINE_HALF);
        use_half = true;
    } else {
        pipeline = ctx.get_pipeline(pipelines::REFINE_MATCHES);
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

        // Determine if we should use MPS mode
        bool use_mps = is_mps_tensor(D11) || is_mps_tensor(D21) || is_mps_tensor(p1);

        // Convert to appropriate type
        if (use_half) {
            D11 = D11.to(torch::kFloat16);
            D21 = D21.to(torch::kFloat16);
        } else {
            D11 = D11.to(torch::kFloat32);
            D21 = D21.to(torch::kFloat32);
        }
        p1 = p1.to(torch::kInt64);

        // Prepare inputs
        MetalBuffer d11_buf = ctx.prepare_input(D11, ctx.input_pool());
        MetalBuffer d21_buf = ctx.prepare_input(D21, ctx.input_pool());
        MetalBuffer p1_buf = ctx.prepare_input(p1, ctx.input_pool());

        // Prepare output
        auto [p1_out_buf, p1_out] = ctx.prepare_output(
            {batch_size, n_points, 2}, torch::kInt64, use_mps);

        struct {
            int batch_size;
            int n_points;
            int H;
            int W;
            int fdim;
            int radius;
            int dilation_max;
        } params = {batch_size, n_points, H, W, fdim, radius, dilation_max};

        id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

        // Execute
        id<MTLCommandBuffer> cmd_buf = ctx.create_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:d11_buf.buffer offset:d11_buf.offset atIndex:0];
        [encoder setBuffer:d21_buf.buffer offset:d21_buf.offset atIndex:1];
        [encoder setBuffer:p1_buf.buffer offset:p1_buf.offset atIndex:2];
        [encoder setBuffer:p1_out_buf.buffer offset:p1_out_buf.offset atIndex:3];
        [encoder setBuffer:params_buf offset:0 atIndex:4];

        // Optimized thread dispatch with aligned threadgroup size
        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;

        NSUInteger alignedWidth = ((n_points + threadWidth - 1) / threadWidth) * threadWidth;
        alignedWidth = MIN(alignedWidth, maxThreads);
        alignedWidth = MAX(alignedWidth, threadWidth);

        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(alignedWidth, (NSUInteger)n_points), 1, 1);

        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        // Finalize output
        ctx.finalize_output(p1_out_buf, p1_out);

        return {p1_out};
    }
}

// ============================================================================
// Batched Operations - Reduced synchronization overhead
// ============================================================================

std::vector<torch::Tensor> iter_proj_and_refine_batched(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh,
    torch::Tensor D11,
    torch::Tensor D21,
    int radius,
    int dilation_max)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* proj_pipeline = ctx.get_pipeline(pipelines::ITER_PROJ);
    auto* refine_pipeline = ctx.get_pipeline(pipelines::REFINE_MATCHES);

    if (!proj_pipeline || !refine_pipeline) {
        TORCH_WARN("Metal batched pipelines not available");
        return {};
    }

    @autoreleasepool {
        // Dimensions for iter_proj
        int batch_size = p_init.size(0);
        int n_points = p_init.size(1);
        int H_rays = rays_img_with_grad.size(1);
        int W_rays = rays_img_with_grad.size(2);

        // Dimensions for refine_matches
        int H_desc = D11.size(1);
        int W_desc = D11.size(2);
        int fdim = D11.size(3);

        // Determine MPS mode
        bool use_mps = is_mps_tensor(rays_img_with_grad) || is_mps_tensor(D11);

        // Ensure types
        rays_img_with_grad = rays_img_with_grad.to(torch::kFloat32);
        pts_3d_norm = pts_3d_norm.to(torch::kFloat32);
        p_init = p_init.to(torch::kFloat32);
        D11 = D11.to(torch::kFloat32);
        D21 = D21.to(torch::kFloat32);

        // Prepare all inputs
        MetalBuffer rays_buf = ctx.prepare_input(rays_img_with_grad, ctx.input_pool());
        MetalBuffer pts_buf = ctx.prepare_input(pts_3d_norm, ctx.input_pool());
        MetalBuffer p_init_buf = ctx.prepare_input(p_init, ctx.input_pool());
        MetalBuffer d11_buf = ctx.prepare_input(D11, ctx.input_pool());
        MetalBuffer d21_buf = ctx.prepare_input(D21, ctx.input_pool());

        // Prepare outputs
        auto [p_proj_buf, p_proj] = ctx.prepare_output(
            {batch_size, n_points, 2}, torch::kFloat32, use_mps);
        auto [conv_buf, converged] = ctx.prepare_output(
            {batch_size, n_points}, torch::kBool, use_mps);

        // Intermediate buffer for projected positions as int64
        auto [p1_buf, p1_int] = ctx.prepare_output(
            {batch_size, n_points, 2}, torch::kInt64, use_mps);

        // Final refined output
        auto [p_refined_buf, p_refined] = ctx.prepare_output(
            {batch_size, n_points, 2}, torch::kInt64, use_mps);

        // Create single batch for both kernels
        CommandBatch batch(ctx);

        // ==== Kernel 1: iter_proj ====
        {
            struct {
                int batch_size;
                int n_points;
                int H;
                int W;
                int max_iter;
                float lambda_init;
                float cost_thresh;
            } params = {batch_size, n_points, H_rays, W_rays, max_iter, lambda_init, cost_thresh};

            id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

            id<MTLComputeCommandEncoder> encoder = batch.add_encoder();
            [encoder setComputePipelineState:proj_pipeline];
            [encoder setBuffer:rays_buf.buffer offset:rays_buf.offset atIndex:0];
            [encoder setBuffer:pts_buf.buffer offset:pts_buf.offset atIndex:1];
            [encoder setBuffer:p_init_buf.buffer offset:p_init_buf.offset atIndex:2];
            [encoder setBuffer:p_proj_buf.buffer offset:p_proj_buf.offset atIndex:3];
            [encoder setBuffer:conv_buf.buffer offset:conv_buf.offset atIndex:4];
            [encoder setBuffer:params_buf offset:0 atIndex:5];

            NSUInteger threadWidth = proj_pipeline.threadExecutionWidth;
            MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
            MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, (NSUInteger)n_points), 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];
        }

        // ==== Convert float positions to int64 (on GPU) ====
        // For now, we need to sync and convert on CPU
        // (A dedicated kernel could optimize this)

        // ==== Kernel 2: refine_matches ====
        {
            struct {
                int batch_size;
                int n_points;
                int H;
                int W;
                int fdim;
                int radius;
                int dilation_max;
            } params = {batch_size, n_points, H_desc, W_desc, fdim, radius, dilation_max};

            id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

            id<MTLComputeCommandEncoder> encoder = batch.add_encoder();
            [encoder setComputePipelineState:refine_pipeline];
            [encoder setBuffer:d11_buf.buffer offset:d11_buf.offset atIndex:0];
            [encoder setBuffer:d21_buf.buffer offset:d21_buf.offset atIndex:1];
            // Use p_proj as input (rounded internally in shader)
            [encoder setBuffer:p_proj_buf.buffer offset:p_proj_buf.offset atIndex:2];
            [encoder setBuffer:p_refined_buf.buffer offset:p_refined_buf.offset atIndex:3];
            [encoder setBuffer:params_buf offset:0 atIndex:4];

            NSUInteger threadWidth = refine_pipeline.threadExecutionWidth;
            MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
            MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, (NSUInteger)n_points), 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];
        }

        // Single commit for both kernels
        batch.commit_and_wait();

        // Finalize outputs
        ctx.finalize_output(p_proj_buf, p_proj);
        ctx.finalize_output(conv_buf, converged);
        ctx.finalize_output(p_refined_buf, p_refined);

        return {p_proj, converged, p_refined};
    }
}

std::vector<torch::Tensor> refine_matches_batched(
    const std::vector<torch::Tensor>& D11_list,
    const std::vector<torch::Tensor>& D21_list,
    const std::vector<torch::Tensor>& p1_list,
    int radius,
    int dilation_max)
{
    if (D11_list.empty() || D11_list.size() != D21_list.size() ||
        D11_list.size() != p1_list.size()) {
        TORCH_WARN("refine_matches_batched: invalid input sizes");
        return {};
    }

    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* pipeline = ctx.get_pipeline(pipelines::REFINE_MATCHES);
    if (!pipeline) {
        TORCH_WARN("Metal refine_matches pipeline not available");
        return {};
    }

    @autoreleasepool {
        size_t n_pairs = D11_list.size();
        std::vector<torch::Tensor> results;
        results.reserve(n_pairs);

        // Prepare all inputs
        std::vector<MetalBuffer> d11_bufs, d21_bufs, p1_bufs;
        std::vector<std::pair<MetalBuffer, torch::Tensor>> outputs;

        d11_bufs.reserve(n_pairs);
        d21_bufs.reserve(n_pairs);
        p1_bufs.reserve(n_pairs);
        outputs.reserve(n_pairs);

        bool use_mps = !D11_list.empty() && is_mps_tensor(D11_list[0]);

        for (size_t i = 0; i < n_pairs; i++) {
            torch::Tensor D11 = D11_list[i].to(torch::kFloat32);
            torch::Tensor D21 = D21_list[i].to(torch::kFloat32);
            torch::Tensor p1 = p1_list[i].to(torch::kInt64);

            d11_bufs.push_back(ctx.prepare_input(D11, ctx.input_pool()));
            d21_bufs.push_back(ctx.prepare_input(D21, ctx.input_pool()));
            p1_bufs.push_back(ctx.prepare_input(p1, ctx.input_pool()));

            int batch_size = p1.size(0);
            int n_points = p1.size(1);
            outputs.push_back(ctx.prepare_output(
                {batch_size, n_points, 2}, torch::kInt64, use_mps));
        }

        // Single batch for all pairs
        CommandBatch batch(ctx);

        for (size_t i = 0; i < n_pairs; i++) {
            torch::Tensor D11 = D11_list[i];
            torch::Tensor p1 = p1_list[i];

            int batch_size = p1.size(0);
            int n_points = p1.size(1);
            int H = D11.size(1);
            int W = D11.size(2);
            int fdim = D11.size(3);

            struct {
                int batch_size;
                int n_points;
                int H;
                int W;
                int fdim;
                int radius;
                int dilation_max;
            } params = {batch_size, n_points, H, W, fdim, radius, dilation_max};

            id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

            id<MTLComputeCommandEncoder> encoder = batch.add_encoder();
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:d11_bufs[i].buffer offset:d11_bufs[i].offset atIndex:0];
            [encoder setBuffer:d21_bufs[i].buffer offset:d21_bufs[i].offset atIndex:1];
            [encoder setBuffer:p1_bufs[i].buffer offset:p1_bufs[i].offset atIndex:2];
            [encoder setBuffer:outputs[i].first.buffer offset:outputs[i].first.offset atIndex:3];
            [encoder setBuffer:params_buf offset:0 atIndex:4];

            NSUInteger threadWidth = pipeline.threadExecutionWidth;
            MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
            MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, (NSUInteger)n_points), 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];
        }

        // Single commit for all kernels
        batch.commit_and_wait();

        // Finalize all outputs
        for (size_t i = 0; i < n_pairs; i++) {
            ctx.finalize_output(outputs[i].first, outputs[i].second);
            results.push_back(outputs[i].second);
        }

        return results;
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
