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
// Gauss-Newton Kernels
// ============================================================================

// Parameter structures matching Metal shaders
struct RayAlignParams {
    int n_edges;
    int n_pts_per_frame;
    float sigma_ray;
    float sigma_dist;
    float C_thresh;
    float Q_thresh;
};

struct CalibProjParams {
    int n_edges;
    int n_pts_per_frame;
    int height;
    int width;
    int pixel_border;
    float z_eps;
    float sigma_pixel;
    float sigma_depth;
    float C_thresh;
    float Q_thresh;
};

struct PoseRetrParams {
    int n_poses;     // Total number of poses
    int num_fix;     // Number of fixed poses to skip
};

// Constants
constexpr int THREADS_GN = 256;
constexpr int POSE_DIM = 7;

std::vector<torch::Tensor> gauss_newton_rays(
    torch::Tensor poses, torch::Tensor points, torch::Tensor confidences,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
    torch::Tensor valid_match, torch::Tensor Q,
    float sigma_ray, float sigma_dist, float C_thresh, float Q_thresh,
    int max_iter, float delta_thresh)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* ray_align_pipeline = ctx.get_pipeline(pipelines::RAY_ALIGN_GN);
    auto* pose_retr_pipeline = ctx.get_pipeline(pipelines::POSE_RETR);

    if (!ray_align_pipeline || !pose_retr_pipeline) {
        TORCH_WARN("Metal gauss_newton_rays pipelines not available");
        return {};
    }

    @autoreleasepool {
        // Ensure correct types
        poses = poses.to(torch::kFloat32).contiguous();
        points = points.to(torch::kFloat32).contiguous();
        confidences = confidences.to(torch::kFloat32).contiguous();
        ii = ii.to(torch::kInt32).contiguous();
        jj = jj.to(torch::kInt32).contiguous();
        idx_ii2jj = idx_ii2jj.to(torch::kInt32).contiguous();
        valid_match = valid_match.to(torch::kBool).contiguous();
        Q = Q.to(torch::kFloat32).contiguous();

        int num_poses = points.size(0);
        int num_points = points.size(1);
        int num_edges = ii.size(0);
        int num_fix = 1;  // Fix first pose

        bool use_mps = is_mps_tensor(poses);

        // Prepare buffers
        MetalBuffer poses_buf = ctx.prepare_input(poses, ctx.input_pool());
        MetalBuffer points_buf = ctx.prepare_input(points, ctx.input_pool());
        MetalBuffer conf_buf = ctx.prepare_input(confidences, ctx.input_pool());
        MetalBuffer ii_buf = ctx.prepare_input(ii, ctx.input_pool());
        MetalBuffer jj_buf = ctx.prepare_input(jj, ctx.input_pool());
        MetalBuffer idx_buf = ctx.prepare_input(idx_ii2jj, ctx.input_pool());
        MetalBuffer valid_buf = ctx.prepare_input(valid_match, ctx.input_pool());
        MetalBuffer Q_buf = ctx.prepare_input(Q, ctx.input_pool());

        // Output buffers for Hessian blocks and gradients
        // Hs: [4, num_edges, 7, 7] - Hii, Hij^T, Hij, Hjj
        // gs: [2, num_edges, 7] - gi, gj
        auto [Hs_buf, Hs] = ctx.prepare_output(
            {4, num_edges, POSE_DIM, POSE_DIM}, torch::kFloat32, use_mps);
        auto [gs_buf, gs] = ctx.prepare_output(
            {2, num_edges, POSE_DIM}, torch::kFloat32, use_mps);

        // Parameters
        RayAlignParams params = {
            num_edges, num_points,
            sigma_ray, sigma_dist,
            C_thresh, Q_thresh
        };
        id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

        // Threadgroup memory size for block reduction
        size_t shared_mem_size = THREADS_GN * sizeof(float);

        // GN iteration loop
        torch::Tensor dx;
        for (int iter = 0; iter < max_iter; iter++) {
            // Zero output buffers
            Hs.zero_();
            gs.zero_();

            // Dispatch ray_align_kernel
            id<MTLCommandBuffer> cmd_buf = ctx.create_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

            [encoder setComputePipelineState:ray_align_pipeline];
            [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
            [encoder setBuffer:points_buf.buffer offset:points_buf.offset atIndex:1];
            [encoder setBuffer:conf_buf.buffer offset:conf_buf.offset atIndex:2];
            [encoder setBuffer:ii_buf.buffer offset:ii_buf.offset atIndex:3];
            [encoder setBuffer:jj_buf.buffer offset:jj_buf.offset atIndex:4];
            [encoder setBuffer:idx_buf.buffer offset:idx_buf.offset atIndex:5];
            [encoder setBuffer:valid_buf.buffer offset:valid_buf.offset atIndex:6];
            [encoder setBuffer:Q_buf.buffer offset:Q_buf.offset atIndex:7];
            [encoder setBuffer:Hs_buf.buffer offset:Hs_buf.offset atIndex:8];
            [encoder setBuffer:gs_buf.buffer offset:gs_buf.offset atIndex:9];
            [encoder setBuffer:params_buf offset:0 atIndex:10];
            [encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0];

            MTLSize grid = MTLSizeMake(num_edges, 1, 1);
            MTLSize threadgroup = MTLSizeMake(THREADS_GN, 1, 1);
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            // Sync outputs to CPU for sparse solve
            ctx.finalize_output(Hs_buf, Hs);
            ctx.finalize_output(gs_buf, gs);

            // Move to CPU for solve
            torch::Tensor Hs_cpu = Hs.to(torch::kCPU);
            torch::Tensor gs_cpu = gs.to(torch::kCPU);
            torch::Tensor ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
            torch::Tensor jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

            // Build sparse system and solve on CPU
            // Using dense solve for now (sparse would require Eigen)
            int n_opt = num_poses - num_fix;
            torch::Tensor H = torch::zeros({n_opt * POSE_DIM, n_opt * POSE_DIM}, torch::kFloat64);
            torch::Tensor b = torch::zeros({n_opt * POSE_DIM}, torch::kFloat64);

            auto ii_acc = ii_cpu.accessor<int64_t, 1>();
            auto jj_acc = jj_cpu.accessor<int64_t, 1>();

            // Accumulate sparse blocks into dense matrix
            for (int e = 0; e < num_edges; e++) {
                int i = ii_acc[e] - num_fix;
                int j = jj_acc[e] - num_fix;

                if (i >= 0) {
                    // Add Hii block
                    H.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM)
                     .slice(1, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        Hs_cpu[0][e].to(torch::kFloat64);

                    // Add gi
                    b.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        gs_cpu[0][e].to(torch::kFloat64);
                }

                if (j >= 0) {
                    // Add Hjj block
                    H.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM)
                     .slice(1, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        Hs_cpu[3][e].to(torch::kFloat64);

                    // Add gj
                    b.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        gs_cpu[1][e].to(torch::kFloat64);
                }

                if (i >= 0 && j >= 0) {
                    // Add Hij and Hij^T blocks
                    H.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM)
                     .slice(1, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        Hs_cpu[1][e].to(torch::kFloat64);

                    H.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM)
                     .slice(1, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        Hs_cpu[2][e].to(torch::kFloat64);
                }
            }

            // Solve: dx = -H^{-1} * b
            // Add damping for stability
            float damping = 1e-4f;
            H.diagonal().add_(damping);

            try {
                auto result = torch::linalg::solve(H, b.unsqueeze(1));
                dx = -result.squeeze(1).to(torch::kFloat32).reshape({n_opt, POSE_DIM});
            } catch (...) {
                dx = torch::zeros({n_opt, POSE_DIM}, torch::kFloat32);
            }

            // Check convergence
            float delta_norm = dx.norm().item<float>();
            if (delta_norm < delta_thresh) {
                break;
            }

            // Apply pose retraction on GPU
            torch::Tensor dx_gpu = use_mps ? dx.to(torch::kMPS) : dx;
            MetalBuffer dx_buf = ctx.prepare_input(dx_gpu, ctx.input_pool());

            PoseRetrParams retr_params = {num_poses, num_fix};
            id<MTLBuffer> retr_params_buf = ctx.create_buffer_with_data(&retr_params, sizeof(retr_params));

            cmd_buf = ctx.create_command_buffer();
            encoder = [cmd_buf computeCommandEncoder];

            [encoder setComputePipelineState:pose_retr_pipeline];
            [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
            [encoder setBuffer:dx_buf.buffer offset:dx_buf.offset atIndex:1];
            [encoder setBuffer:retr_params_buf offset:0 atIndex:2];

            // Dispatch one thread per non-fixed pose
            int n_opt_dispatch = num_poses - num_fix;
            grid = MTLSizeMake((n_opt_dispatch + 255) / 256, 1, 1);
            threadgroup = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
        }

        return {dx};
    }
}

std::vector<torch::Tensor> gauss_newton_calib(
    torch::Tensor poses, torch::Tensor points, torch::Tensor confidences, torch::Tensor K,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
    torch::Tensor valid_match, torch::Tensor Q,
    int height, int width, int pixel_border, float z_eps,
    float sigma_pixel, float sigma_depth, float C_thresh, float Q_thresh,
    int max_iter, float delta_thresh)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* calib_proj_pipeline = ctx.get_pipeline(pipelines::CALIB_PROJ_GN);
    auto* pose_retr_pipeline = ctx.get_pipeline(pipelines::POSE_RETR);

    if (!calib_proj_pipeline || !pose_retr_pipeline) {
        TORCH_WARN("Metal gauss_newton_calib pipelines not available");
        return {};
    }

    @autoreleasepool {
        // Ensure correct types
        poses = poses.to(torch::kFloat32).contiguous();
        points = points.to(torch::kFloat32).contiguous();
        confidences = confidences.to(torch::kFloat32).contiguous();
        K = K.to(torch::kFloat32).contiguous();
        ii = ii.to(torch::kInt32).contiguous();
        jj = jj.to(torch::kInt32).contiguous();
        idx_ii2jj = idx_ii2jj.to(torch::kInt32).contiguous();
        valid_match = valid_match.to(torch::kBool).contiguous();
        Q = Q.to(torch::kFloat32).contiguous();

        int num_poses = points.size(0);
        int num_points = points.size(1);
        int num_edges = ii.size(0);
        int num_fix = 1;

        bool use_mps = is_mps_tensor(poses);

        // Prepare buffers
        MetalBuffer poses_buf = ctx.prepare_input(poses, ctx.input_pool());
        MetalBuffer points_buf = ctx.prepare_input(points, ctx.input_pool());
        MetalBuffer conf_buf = ctx.prepare_input(confidences, ctx.input_pool());
        MetalBuffer K_buf = ctx.prepare_input(K, ctx.input_pool());
        MetalBuffer ii_buf = ctx.prepare_input(ii, ctx.input_pool());
        MetalBuffer jj_buf = ctx.prepare_input(jj, ctx.input_pool());
        MetalBuffer idx_buf = ctx.prepare_input(idx_ii2jj, ctx.input_pool());
        MetalBuffer valid_buf = ctx.prepare_input(valid_match, ctx.input_pool());
        MetalBuffer Q_buf = ctx.prepare_input(Q, ctx.input_pool());

        auto [Hs_buf, Hs] = ctx.prepare_output(
            {4, num_edges, POSE_DIM, POSE_DIM}, torch::kFloat32, use_mps);
        auto [gs_buf, gs] = ctx.prepare_output(
            {2, num_edges, POSE_DIM}, torch::kFloat32, use_mps);

        CalibProjParams params = {
            num_edges, num_points,
            height, width, pixel_border, z_eps,
            sigma_pixel, sigma_depth,
            C_thresh, Q_thresh
        };
        id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

        size_t shared_mem_size = THREADS_GN * sizeof(float);

        torch::Tensor dx;
        for (int iter = 0; iter < max_iter; iter++) {
            Hs.zero_();
            gs.zero_();

            id<MTLCommandBuffer> cmd_buf = ctx.create_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

            [encoder setComputePipelineState:calib_proj_pipeline];
            [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
            [encoder setBuffer:points_buf.buffer offset:points_buf.offset atIndex:1];
            [encoder setBuffer:conf_buf.buffer offset:conf_buf.offset atIndex:2];
            [encoder setBuffer:K_buf.buffer offset:K_buf.offset atIndex:3];
            [encoder setBuffer:ii_buf.buffer offset:ii_buf.offset atIndex:4];
            [encoder setBuffer:jj_buf.buffer offset:jj_buf.offset atIndex:5];
            [encoder setBuffer:idx_buf.buffer offset:idx_buf.offset atIndex:6];
            [encoder setBuffer:valid_buf.buffer offset:valid_buf.offset atIndex:7];
            [encoder setBuffer:Q_buf.buffer offset:Q_buf.offset atIndex:8];
            [encoder setBuffer:Hs_buf.buffer offset:Hs_buf.offset atIndex:9];
            [encoder setBuffer:gs_buf.buffer offset:gs_buf.offset atIndex:10];
            [encoder setBuffer:params_buf offset:0 atIndex:11];
            [encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0];

            MTLSize grid = MTLSizeMake(num_edges, 1, 1);
            MTLSize threadgroup = MTLSizeMake(THREADS_GN, 1, 1);
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            ctx.finalize_output(Hs_buf, Hs);
            ctx.finalize_output(gs_buf, gs);

            // CPU solve (same as ray_align)
            torch::Tensor Hs_cpu = Hs.to(torch::kCPU);
            torch::Tensor gs_cpu = gs.to(torch::kCPU);
            torch::Tensor ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
            torch::Tensor jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

            int n_opt = num_poses - num_fix;
            torch::Tensor H = torch::zeros({n_opt * POSE_DIM, n_opt * POSE_DIM}, torch::kFloat64);
            torch::Tensor b = torch::zeros({n_opt * POSE_DIM}, torch::kFloat64);

            auto ii_acc = ii_cpu.accessor<int64_t, 1>();
            auto jj_acc = jj_cpu.accessor<int64_t, 1>();

            for (int e = 0; e < num_edges; e++) {
                int i = ii_acc[e] - num_fix;
                int j = jj_acc[e] - num_fix;

                if (i >= 0) {
                    H.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM)
                     .slice(1, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        Hs_cpu[0][e].to(torch::kFloat64);
                    b.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        gs_cpu[0][e].to(torch::kFloat64);
                }

                if (j >= 0) {
                    H.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM)
                     .slice(1, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        Hs_cpu[3][e].to(torch::kFloat64);
                    b.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        gs_cpu[1][e].to(torch::kFloat64);
                }

                if (i >= 0 && j >= 0) {
                    H.slice(0, i * POSE_DIM, (i + 1) * POSE_DIM)
                     .slice(1, j * POSE_DIM, (j + 1) * POSE_DIM) +=
                        Hs_cpu[1][e].to(torch::kFloat64);
                    H.slice(0, j * POSE_DIM, (j + 1) * POSE_DIM)
                     .slice(1, i * POSE_DIM, (i + 1) * POSE_DIM) +=
                        Hs_cpu[2][e].to(torch::kFloat64);
                }
            }

            float damping = 1e-4f;
            H.diagonal().add_(damping);

            try {
                auto result = torch::linalg::solve(H, b.unsqueeze(1));
                dx = -result.squeeze(1).to(torch::kFloat32).reshape({n_opt, POSE_DIM});
            } catch (...) {
                dx = torch::zeros({n_opt, POSE_DIM}, torch::kFloat32);
            }

            float delta_norm = dx.norm().item<float>();
            if (delta_norm < delta_thresh) {
                break;
            }

            torch::Tensor dx_gpu = use_mps ? dx.to(torch::kMPS) : dx;
            MetalBuffer dx_buf = ctx.prepare_input(dx_gpu, ctx.input_pool());

            PoseRetrParams retr_params = {num_poses, num_fix};
            id<MTLBuffer> retr_params_buf = ctx.create_buffer_with_data(&retr_params, sizeof(retr_params));

            cmd_buf = ctx.create_command_buffer();
            encoder = [cmd_buf computeCommandEncoder];

            [encoder setComputePipelineState:pose_retr_pipeline];
            [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
            [encoder setBuffer:dx_buf.buffer offset:dx_buf.offset atIndex:1];
            [encoder setBuffer:retr_params_buf offset:0 atIndex:2];

            // Dispatch one thread per non-fixed pose
            int n_opt_dispatch = num_poses - num_fix;
            grid = MTLSizeMake((n_opt_dispatch + 255) / 256, 1, 1);
            threadgroup = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];
        }

        return {dx};
    }
}

} // namespace metal_backend
