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

        // Optimized thread dispatch
        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, (NSUInteger)n_points), 1, 1);

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

        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(n_points, batch_size, 1);
        MTLSize threadgroup = MTLSizeMake(MIN(threadWidth, (NSUInteger)n_points), 1, 1);

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
    auto* refine_pipeline = ctx.get_pipeline(pipelines::REFINE_FROM_FLOAT);  // Uses float input!

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
        // Final refined output (refine_from_float reads float directly, no conversion needed)
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

        // ==== Kernel 2: refine_matches (from float positions) ====
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
// GN Helper: Assemble and solve linear system using PyTorch MPS
// Replaces Eigen SparseBlock with pure PyTorch operations
// ============================================================================

namespace {

// Structure for Gauss-Newton parameters passed to Metal
struct GNParams {
    uint32_t num_points;
    uint32_t num_edges;
    float sigma_ray;
    float sigma_dist;
    float C_thresh;
    float Q_thresh;
};

struct CalibParams {
    uint32_t num_points;
    uint32_t num_edges;
    uint32_t height;
    uint32_t width;
    uint32_t pixel_border;
    float z_eps;
    float sigma_pixel;
    float sigma_depth;
    float C_thresh;
    float Q_thresh;
};

// Assemble sparse system from Hessian blocks and solve using PyTorch
// Replaces Eigen SparseBlock with dense solve (faster for small systems on GPU)
torch::Tensor assemble_and_solve(
    torch::Tensor Hs,           // [4, num_edges, 7, 7]
    torch::Tensor gs,           // [2, num_edges, 7]
    torch::Tensor ii_opt,       // [num_edges] - optimized indices for i
    torch::Tensor jj_opt,       // [num_edges] - optimized indices for j
    int num_poses_opt,          // Number of poses to optimize (excluding fixed)
    float lm_damping = 1e-4f)   // Levenberg-Marquardt damping
{
    const int pose_dim = 7;
    const int num_edges = ii_opt.size(0);
    auto opts = Hs.options();

    // Create dense Hessian [num_poses_opt * 7, num_poses_opt * 7]
    int system_size = num_poses_opt * pose_dim;
    torch::Tensor H = torch::zeros({system_size, system_size}, opts);
    torch::Tensor b = torch::zeros({system_size}, opts);

    // Hs layout: [Hii, Hij, Hji, Hjj] each [num_edges, 7, 7]
    torch::Tensor Hii = Hs[0];  // [E, 7, 7]
    torch::Tensor Hij = Hs[1];  // [E, 7, 7]
    torch::Tensor Hji = Hs[2];  // [E, 7, 7]
    torch::Tensor Hjj = Hs[3];  // [E, 7, 7]

    // gs layout: [gi, gj] each [num_edges, 7]
    torch::Tensor gi = gs[0];   // [E, 7]
    torch::Tensor gj = gs[1];   // [E, 7]

    // Scatter-add to build system (using index_add for efficiency)
    auto ii_opt_cpu = ii_opt.to(torch::kCPU).to(torch::kInt64);
    auto jj_opt_cpu = jj_opt.to(torch::kCPU).to(torch::kInt64);
    auto ii_acc = ii_opt_cpu.accessor<int64_t, 1>();
    auto jj_acc = jj_opt_cpu.accessor<int64_t, 1>();

    // Move Hs, gs to CPU for assembly (small data)
    auto Hii_cpu = Hii.to(torch::kCPU);
    auto Hij_cpu = Hij.to(torch::kCPU);
    auto Hji_cpu = Hji.to(torch::kCPU);
    auto Hjj_cpu = Hjj.to(torch::kCPU);
    auto gi_cpu = gi.to(torch::kCPU);
    auto gj_cpu = gj.to(torch::kCPU);

    auto H_cpu = H.to(torch::kCPU);
    auto b_cpu = b.to(torch::kCPU);

    for (int e = 0; e < num_edges; e++) {
        int64_t i = ii_acc[e];
        int64_t j = jj_acc[e];

        // Skip fixed poses (negative indices)
        if (i >= 0) {
            int i_start = i * pose_dim;
            // Add Hii block
            H_cpu.index({torch::indexing::Slice(i_start, i_start + pose_dim),
                         torch::indexing::Slice(i_start, i_start + pose_dim)}) += Hii_cpu[e];
            // Add gradient
            b_cpu.index({torch::indexing::Slice(i_start, i_start + pose_dim)}) += gi_cpu[e];
        }

        if (j >= 0) {
            int j_start = j * pose_dim;
            // Add Hjj block
            H_cpu.index({torch::indexing::Slice(j_start, j_start + pose_dim),
                         torch::indexing::Slice(j_start, j_start + pose_dim)}) += Hjj_cpu[e];
            // Add gradient
            b_cpu.index({torch::indexing::Slice(j_start, j_start + pose_dim)}) += gj_cpu[e];
        }

        if (i >= 0 && j >= 0) {
            int i_start = i * pose_dim;
            int j_start = j * pose_dim;
            // Add off-diagonal blocks (symmetric)
            H_cpu.index({torch::indexing::Slice(i_start, i_start + pose_dim),
                         torch::indexing::Slice(j_start, j_start + pose_dim)}) += Hij_cpu[e];
            H_cpu.index({torch::indexing::Slice(j_start, j_start + pose_dim),
                         torch::indexing::Slice(i_start, i_start + pose_dim)}) += Hji_cpu[e];
        }
    }

    // Add LM damping to diagonal
    for (int i = 0; i < system_size; i++) {
        H_cpu[i][i] += lm_damping * H_cpu[i][i] + 1e-6f;
    }

    // Move to MPS for solve (if available)
    torch::Device solve_device = torch::kCPU;
    if (torch::mps::is_available()) {
        solve_device = torch::kMPS;
    }

    H = H_cpu.to(solve_device);
    b = b_cpu.to(solve_device);

    // Solve H * dx = -b using Cholesky
    torch::Tensor dx;
    try {
        // Use Cholesky for symmetric positive definite (ATen API)
        torch::Tensor L = at::linalg_cholesky(H);
        dx = at::cholesky_solve(b.unsqueeze(1), L).squeeze(1);
        dx = -dx;  // dx = -H^{-1} * b
    } catch (...) {
        // Fallback to general solve (ATen API)
        dx = -at::linalg_solve(H, b);
    }

    // Reshape to [num_poses_opt, 7]
    return dx.reshape({num_poses_opt, pose_dim});
}

// Create optimized indices (excluding fixed poses)
std::pair<torch::Tensor, torch::Tensor> create_opt_indices(
    torch::Tensor ii, torch::Tensor jj, int num_fix)
{
    // Get unique keyframe indices
    torch::Tensor all_idx = torch::cat({ii, jj});
    torch::Tensor unique_idx = std::get<0>(torch::_unique(all_idx, true));

    // Create mapping: original index -> optimized index (excluding first num_fix)
    torch::Tensor ii_opt = torch::searchsorted(unique_idx, ii) - num_fix;
    torch::Tensor jj_opt = torch::searchsorted(unique_idx, jj) - num_fix;

    return {ii_opt, jj_opt};
}

} // anonymous namespace

// ============================================================================
// Gauss-Newton Rays - Metal kernel + PyTorch solve
// ============================================================================

std::vector<torch::Tensor> gauss_newton_rays(
    torch::Tensor poses, torch::Tensor points, torch::Tensor confidences,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
    torch::Tensor valid_match, torch::Tensor Q,
    float sigma_ray, float sigma_dist, float C_thresh, float Q_thresh,
    int max_iter, float delta_thresh)
{
    auto& ctx = MetalContext::instance();
    if (!ctx.initialize()) return {};

    auto* ray_pipeline = ctx.get_pipeline(pipelines::RAY_ALIGN);
    auto* retr_pipeline = ctx.get_pipeline(pipelines::POSE_RETR);

    if (!ray_pipeline || !retr_pipeline) {
        TORCH_WARN("gauss_newton_rays: Metal pipelines not available");
        return {};
    }

    @autoreleasepool {
        const int num_edges = ii.size(0);
        const int num_poses = points.size(0);
        const int num_points = points.size(1);
        const int pose_dim = 7;
        const int num_fix = 1;

        // Determine device
        bool use_mps = is_mps_tensor(poses);
        auto opts = poses.options();

        // Ensure types and contiguity
        poses = poses.to(torch::kFloat32).contiguous();
        points = points.to(torch::kFloat32).contiguous();
        confidences = confidences.to(torch::kFloat32).contiguous();
        ii = ii.to(torch::kInt64).contiguous();
        jj = jj.to(torch::kInt64).contiguous();
        idx_ii2jj = idx_ii2jj.to(torch::kInt64).contiguous();
        valid_match = valid_match.to(torch::kBool).contiguous();
        Q = Q.to(torch::kFloat32).contiguous();

        // Create optimized indices
        auto [ii_opt, jj_opt] = create_opt_indices(ii, jj, num_fix);
        int num_poses_opt = num_poses - num_fix;

        // Allocate Hessian and gradient buffers
        torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
        torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);
        torch::Tensor dx;

        for (int itr = 0; itr < max_iter; itr++) {
            // Reset buffers
            Hs.zero_();
            gs.zero_();

            // Prepare inputs for Metal kernel
            MetalBuffer poses_buf = ctx.prepare_input(poses, ctx.input_pool());
            MetalBuffer points_buf = ctx.prepare_input(points, ctx.input_pool());
            MetalBuffer conf_buf = ctx.prepare_input(confidences, ctx.input_pool());
            MetalBuffer ii_buf = ctx.prepare_input(ii, ctx.input_pool());
            MetalBuffer jj_buf = ctx.prepare_input(jj, ctx.input_pool());
            MetalBuffer idx_buf = ctx.prepare_input(idx_ii2jj, ctx.input_pool());
            MetalBuffer valid_buf = ctx.prepare_input(valid_match, ctx.input_pool());
            MetalBuffer Q_buf = ctx.prepare_input(Q, ctx.input_pool());
            MetalBuffer Hs_buf = ctx.prepare_input(Hs, ctx.output_pool());
            MetalBuffer gs_buf = ctx.prepare_input(gs, ctx.output_pool());

            // Params
            GNParams params = {
                (uint32_t)num_points,
                (uint32_t)num_edges,
                sigma_ray,
                sigma_dist,
                C_thresh,
                Q_thresh
            };
            id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

            // Execute ray_align kernel
            CommandBatch batch(ctx);
            {
                id<MTLComputeCommandEncoder> encoder = batch.add_encoder();
                [encoder setComputePipelineState:ray_pipeline];
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

                // Shared memory for reduction
                [encoder setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];

                // One threadgroup per edge
                MTLSize grid = MTLSizeMake(256 * num_edges, 1, 1);
                MTLSize threadgroup = MTLSizeMake(256, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                [encoder endEncoding];
            }
            batch.commit_and_wait();

            // Finalize Hs, gs
            ctx.finalize_output(Hs_buf, Hs);
            ctx.finalize_output(gs_buf, gs);

            // Solve linear system using PyTorch
            dx = assemble_and_solve(Hs, gs, ii_opt, jj_opt, num_poses_opt);

            // Apply retraction using Metal kernel
            if (use_mps) {
                dx = dx.to(torch::kMPS).contiguous();
            }
            MetalBuffer dx_buf = ctx.prepare_input(dx, ctx.input_pool());

            uint32_t num_poses_u = (uint32_t)num_poses;
            uint32_t num_fix_u = (uint32_t)num_fix;

            id<MTLBuffer> num_poses_buf = ctx.create_buffer_with_data(&num_poses_u, sizeof(uint32_t));
            id<MTLBuffer> num_fix_buf = ctx.create_buffer_with_data(&num_fix_u, sizeof(uint32_t));

            CommandBatch retr_batch(ctx);
            {
                id<MTLComputeCommandEncoder> encoder = retr_batch.add_encoder();
                [encoder setComputePipelineState:retr_pipeline];
                [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
                [encoder setBuffer:dx_buf.buffer offset:dx_buf.offset atIndex:1];
                [encoder setBuffer:num_poses_buf offset:0 atIndex:2];
                [encoder setBuffer:num_fix_buf offset:0 atIndex:3];

                MTLSize grid = MTLSizeMake(num_poses_opt, 1, 1);
                MTLSize threadgroup = MTLSizeMake(MIN(64, (NSUInteger)num_poses_opt), 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                [encoder endEncoding];
            }
            retr_batch.commit_and_wait();

            // Update poses
            ctx.finalize_output(poses_buf, poses);

            // Check convergence
            float delta_norm = dx.norm().item<float>();
            if (delta_norm < delta_thresh) {
                break;
            }
        }

        return {dx};
    }
}

// ============================================================================
// Gauss-Newton Calib - Metal kernel + PyTorch solve
// ============================================================================

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

    auto* calib_pipeline = ctx.get_pipeline(pipelines::CALIB_PROJ);
    auto* retr_pipeline = ctx.get_pipeline(pipelines::POSE_RETR);

    if (!calib_pipeline || !retr_pipeline) {
        TORCH_WARN("gauss_newton_calib: Metal pipelines not available");
        return {};
    }

    @autoreleasepool {
        const int num_edges = ii.size(0);
        const int num_poses = points.size(0);
        const int num_points = points.size(1);
        const int pose_dim = 7;
        const int num_fix = 1;

        bool use_mps = is_mps_tensor(poses);
        auto opts = poses.options();

        // Ensure types
        poses = poses.to(torch::kFloat32).contiguous();
        points = points.to(torch::kFloat32).contiguous();
        confidences = confidences.to(torch::kFloat32).contiguous();
        K = K.to(torch::kFloat32).contiguous();
        ii = ii.to(torch::kInt64).contiguous();
        jj = jj.to(torch::kInt64).contiguous();
        idx_ii2jj = idx_ii2jj.to(torch::kInt64).contiguous();
        valid_match = valid_match.to(torch::kBool).contiguous();
        Q = Q.to(torch::kFloat32).contiguous();

        auto [ii_opt, jj_opt] = create_opt_indices(ii, jj, num_fix);
        int num_poses_opt = num_poses - num_fix;

        torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
        torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);
        torch::Tensor dx;

        for (int itr = 0; itr < max_iter; itr++) {
            Hs.zero_();
            gs.zero_();

            MetalBuffer poses_buf = ctx.prepare_input(poses, ctx.input_pool());
            MetalBuffer points_buf = ctx.prepare_input(points, ctx.input_pool());
            MetalBuffer conf_buf = ctx.prepare_input(confidences, ctx.input_pool());
            MetalBuffer K_buf = ctx.prepare_input(K, ctx.input_pool());
            MetalBuffer ii_buf = ctx.prepare_input(ii, ctx.input_pool());
            MetalBuffer jj_buf = ctx.prepare_input(jj, ctx.input_pool());
            MetalBuffer idx_buf = ctx.prepare_input(idx_ii2jj, ctx.input_pool());
            MetalBuffer valid_buf = ctx.prepare_input(valid_match, ctx.input_pool());
            MetalBuffer Q_buf = ctx.prepare_input(Q, ctx.input_pool());
            MetalBuffer Hs_buf = ctx.prepare_input(Hs, ctx.output_pool());
            MetalBuffer gs_buf = ctx.prepare_input(gs, ctx.output_pool());

            CalibParams params = {
                (uint32_t)num_points,
                (uint32_t)num_edges,
                (uint32_t)height,
                (uint32_t)width,
                (uint32_t)pixel_border,
                z_eps,
                sigma_pixel,
                sigma_depth,
                C_thresh,
                Q_thresh
            };
            id<MTLBuffer> params_buf = ctx.create_buffer_with_data(&params, sizeof(params));

            CommandBatch batch(ctx);
            {
                id<MTLComputeCommandEncoder> encoder = batch.add_encoder();
                [encoder setComputePipelineState:calib_pipeline];
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

                [encoder setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];

                MTLSize grid = MTLSizeMake(256 * num_edges, 1, 1);
                MTLSize threadgroup = MTLSizeMake(256, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                [encoder endEncoding];
            }
            batch.commit_and_wait();

            ctx.finalize_output(Hs_buf, Hs);
            ctx.finalize_output(gs_buf, gs);

            dx = assemble_and_solve(Hs, gs, ii_opt, jj_opt, num_poses_opt);

            if (use_mps) {
                dx = dx.to(torch::kMPS).contiguous();
            }
            MetalBuffer dx_buf = ctx.prepare_input(dx, ctx.input_pool());

            uint32_t num_poses_u = (uint32_t)num_poses;
            uint32_t num_fix_u = (uint32_t)num_fix;

            id<MTLBuffer> num_poses_buf = ctx.create_buffer_with_data(&num_poses_u, sizeof(uint32_t));
            id<MTLBuffer> num_fix_buf = ctx.create_buffer_with_data(&num_fix_u, sizeof(uint32_t));

            CommandBatch retr_batch(ctx);
            {
                id<MTLComputeCommandEncoder> encoder = retr_batch.add_encoder();
                [encoder setComputePipelineState:retr_pipeline];
                [encoder setBuffer:poses_buf.buffer offset:poses_buf.offset atIndex:0];
                [encoder setBuffer:dx_buf.buffer offset:dx_buf.offset atIndex:1];
                [encoder setBuffer:num_poses_buf offset:0 atIndex:2];
                [encoder setBuffer:num_fix_buf offset:0 atIndex:3];

                MTLSize grid = MTLSizeMake(num_poses_opt, 1, 1);
                MTLSize threadgroup = MTLSizeMake(MIN(64, (NSUInteger)num_poses_opt), 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
                [encoder endEncoding];
            }
            retr_batch.commit_and_wait();

            ctx.finalize_output(poses_buf, poses);

            float delta_norm = dx.norm().item<float>();
            if (delta_norm < delta_thresh) {
                break;
            }
        }

        return {dx};
    }
}

} // namespace metal_backend
