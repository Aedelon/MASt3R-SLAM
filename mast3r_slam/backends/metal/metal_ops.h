// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
#pragma once

#include <torch/extension.h>
#include <vector>

namespace metal_backend {

// Initialize Metal device and compile shaders
bool initialize();

// Check if Metal is available
bool is_available();

// Matching kernels
std::vector<torch::Tensor> iter_proj(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh);

std::vector<torch::Tensor> refine_matches(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    int radius,
    int dilation_max);

// ============================================================================
// Batched Operations - Reduced synchronization overhead
// ============================================================================

/**
 * Batched iter_proj + refine_matches in single command buffer.
 * Avoids 2 CPU-GPU synchronization points.
 *
 * Returns: [p_proj, converged, p_refined]
 */
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
    int dilation_max);

/**
 * Multiple refine_matches in single command buffer.
 * For processing multiple frame pairs efficiently.
 *
 * Returns: vector of refined positions for each pair
 */
std::vector<torch::Tensor> refine_matches_batched(
    const std::vector<torch::Tensor>& D11_list,
    const std::vector<torch::Tensor>& D21_list,
    const std::vector<torch::Tensor>& p1_list,
    int radius,
    int dilation_max);

// Gauss-Newton kernels
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
    float delta_thresh);

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
    float delta_thresh);

} // namespace metal_backend
