// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * CPU matching kernels with OpenMP parallelization and SIMD optimizations.
 */

#include "matching.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <limits>

// SIMD headers
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

namespace cpu_backend {

// ============================================================================
// Helper functions
// ============================================================================

inline bool inside_image(int u, int v, int W, int H) {
    return v >= 0 && v < H && u >= 0 && u < W;
}

inline float clamp(float x, float min_val, float max_val) {
    return std::min(std::max(x, min_val), max_val);
}

inline void normalize3(float* v) {
    float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    float inv_norm = 1.0f / (norm + 1e-12f);
    v[0] *= inv_norm;
    v[1] *= inv_norm;
    v[2] *= inv_norm;
}

// ============================================================================
// Iterative Projection Kernel
// ============================================================================

std::vector<torch::Tensor> iter_proj(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    int max_iter,
    float lambda_init,
    float cost_thresh)
{
    CHECK_CONTIGUOUS(rays_img_with_grad);
    CHECK_CONTIGUOUS(pts_3d_norm);
    CHECK_CONTIGUOUS(p_init);

    const int batch_size = p_init.size(0);
    const int n_points = p_init.size(1);
    const int H = rays_img_with_grad.size(1);
    const int W = rays_img_with_grad.size(2);

    auto opts = p_init.options();
    torch::Tensor p_new = torch::zeros({batch_size, n_points, 2}, opts);
    torch::Tensor converged = torch::zeros({batch_size, n_points}, opts.dtype(torch::kBool));

    auto rays_acc = rays_img_with_grad.accessor<float, 4>();
    auto pts_acc = pts_3d_norm.accessor<float, 3>();
    auto p_init_acc = p_init.accessor<float, 3>();
    auto p_new_acc = p_new.accessor<float, 3>();
    auto conv_acc = converged.accessor<bool, 2>();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < n_points; n++) {
            float u = p_init_acc[b][n][0];
            float v = p_init_acc[b][n][1];

            u = clamp(u, 1.0f, (float)(W - 2));
            v = clamp(v, 1.0f, (float)(H - 2));

            float r[3], gx[3], gy[3], err[3];
            float lambda = lambda_init;

            for (int iter = 0; iter < max_iter; iter++) {
                int u11 = (int)std::floor(u);
                int v11 = (int)std::floor(v);
                float du = u - u11;
                float dv = v - v11;

                float w11 = du * dv;
                float w12 = (1.0f - du) * dv;
                float w21 = du * (1.0f - dv);
                float w22 = (1.0f - du) * (1.0f - dv);

                for (int j = 0; j < 3; j++) {
                    r[j] = w11 * rays_acc[b][v11+1][u11+1][j] +
                           w12 * rays_acc[b][v11+1][u11][j] +
                           w21 * rays_acc[b][v11][u11+1][j] +
                           w22 * rays_acc[b][v11][u11][j];
                }
                for (int j = 0; j < 3; j++) {
                    gx[j] = w11 * rays_acc[b][v11+1][u11+1][j+3] +
                            w12 * rays_acc[b][v11+1][u11][j+3] +
                            w21 * rays_acc[b][v11][u11+1][j+3] +
                            w22 * rays_acc[b][v11][u11][j+3];
                }
                for (int j = 0; j < 3; j++) {
                    gy[j] = w11 * rays_acc[b][v11+1][u11+1][j+6] +
                            w12 * rays_acc[b][v11+1][u11][j+6] +
                            w21 * rays_acc[b][v11][u11+1][j+6] +
                            w22 * rays_acc[b][v11][u11][j+6];
                }

                normalize3(r);

                for (int j = 0; j < 3; j++) {
                    err[j] = r[j] - pts_acc[b][n][j];
                }
                float cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

                float A00 = gx[0]*gx[0] + gx[1]*gx[1] + gx[2]*gx[2] + lambda;
                float A01 = gx[0]*gy[0] + gx[1]*gy[1] + gx[2]*gy[2];
                float A11 = gy[0]*gy[0] + gy[1]*gy[1] + gy[2]*gy[2] + lambda;
                float b0 = -(err[0]*gx[0] + err[1]*gx[1] + err[2]*gx[2]);
                float b1 = -(err[0]*gy[0] + err[1]*gy[1] + err[2]*gy[2]);

                float det_inv = 1.0f / (A00*A11 - A01*A01 + 1e-12f);
                float delta_u = det_inv * (A11*b0 - A01*b1);
                float delta_v = det_inv * (-A01*b0 + A00*b1);

                float u_new = clamp(u + delta_u, 1.0f, (float)(W - 2));
                float v_new = clamp(v + delta_v, 1.0f, (float)(H - 2));

                u11 = (int)std::floor(u_new);
                v11 = (int)std::floor(v_new);
                du = u_new - u11;
                dv = v_new - v11;
                w11 = du * dv;
                w12 = (1.0f - du) * dv;
                w21 = du * (1.0f - dv);
                w22 = (1.0f - du) * (1.0f - dv);

                for (int j = 0; j < 3; j++) {
                    r[j] = w11 * rays_acc[b][v11+1][u11+1][j] +
                           w12 * rays_acc[b][v11+1][u11][j] +
                           w21 * rays_acc[b][v11][u11+1][j] +
                           w22 * rays_acc[b][v11][u11][j];
                }
                normalize3(r);

                for (int j = 0; j < 3; j++) {
                    err[j] = r[j] - pts_acc[b][n][j];
                }
                float new_cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

                if (new_cost < cost) {
                    u = u_new;
                    v = v_new;
                    lambda *= 0.1f;
                    conv_acc[b][n] = new_cost < cost_thresh;
                } else {
                    lambda *= 10.0f;
                    conv_acc[b][n] = cost < cost_thresh;
                }
            }

            p_new_acc[b][n][0] = u;
            p_new_acc[b][n][1] = v;
        }
    }

    return {p_new, converged};
}

// ============================================================================
// Refine Matches Kernel
// ============================================================================

std::vector<torch::Tensor> refine_matches(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    int radius,
    int dilation_max)
{
    CHECK_CONTIGUOUS(D11);
    CHECK_CONTIGUOUS(D21);
    CHECK_CONTIGUOUS(p1);

    D11 = D11.to(torch::kFloat32);
    D21 = D21.to(torch::kFloat32);

    const int batch_size = p1.size(0);
    const int n_points = p1.size(1);
    const int H = D11.size(1);
    const int W = D11.size(2);
    const int fdim = D11.size(3);

    auto opts = p1.options();
    torch::Tensor p1_new = torch::zeros({batch_size, n_points, 2}, opts);

    auto D11_acc = D11.accessor<float, 4>();
    auto D21_acc = D21.accessor<float, 3>();
    auto p1_acc = p1.accessor<int64_t, 3>();
    auto p1_new_acc = p1_new.accessor<int64_t, 3>();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < n_points; n++) {
            int64_t u0 = p1_acc[b][n][0];
            int64_t v0 = p1_acc[b][n][1];

            float max_score = -std::numeric_limits<float>::max();
            int64_t u_new = u0;
            int64_t v_new = v0;

            for (int d = dilation_max; d > 0; d--) {
                const int rd = radius * d;
                const int diam = 2 * rd + 1;

                for (int i = 0; i < diam; i += d) {
                    for (int j = 0; j < diam; j += d) {
                        const int64_t u = u0 - rd + i;
                        const int64_t v = v0 - rd + j;

                        if (inside_image(u, v, W, H)) {
                            float score = 0.0f;

                            #if USE_AVX2
                            __m256 sum = _mm256_setzero_ps();
                            int k = 0;
                            for (; k + 8 <= fdim; k += 8) {
                                __m256 a = _mm256_loadu_ps(&D21_acc[b][n][k]);
                                __m256 b_vec = _mm256_loadu_ps(&D11_acc[b][v][u][k]);
                                sum = _mm256_fmadd_ps(a, b_vec, sum);
                            }
                            __m128 hi = _mm256_extractf128_ps(sum, 1);
                            __m128 lo = _mm256_castps256_ps128(sum);
                            __m128 sum128 = _mm_add_ps(hi, lo);
                            sum128 = _mm_hadd_ps(sum128, sum128);
                            sum128 = _mm_hadd_ps(sum128, sum128);
                            score = _mm_cvtss_f32(sum128);
                            for (; k < fdim; k++) {
                                score += D21_acc[b][n][k] * D11_acc[b][v][u][k];
                            }
                            #elif USE_NEON
                            float32x4_t sum_vec = vdupq_n_f32(0.0f);
                            int k = 0;
                            for (; k + 4 <= fdim; k += 4) {
                                float32x4_t a = vld1q_f32(&D21_acc[b][n][k]);
                                float32x4_t b_vec = vld1q_f32(&D11_acc[b][v][u][k]);
                                sum_vec = vmlaq_f32(sum_vec, a, b_vec);
                            }
                            float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                            score = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
                            for (; k < fdim; k++) {
                                score += D21_acc[b][n][k] * D11_acc[b][v][u][k];
                            }
                            #else
                            for (int k = 0; k < fdim; k++) {
                                score += D21_acc[b][n][k] * D11_acc[b][v][u][k];
                            }
                            #endif

                            if (score > max_score) {
                                max_score = score;
                                u_new = u;
                                v_new = v;
                            }
                        }
                    }
                }
                u0 = u_new;
                v0 = v_new;
            }

            p1_new_acc[b][n][0] = u_new;
            p1_new_acc[b][n][1] = v_new;
        }
    }

    return {p1_new};
}

} // namespace cpu_backend
