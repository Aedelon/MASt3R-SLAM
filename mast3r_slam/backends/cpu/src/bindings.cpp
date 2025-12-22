// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Python bindings for CPU backend.
 */

#include "gn.h"
#include "matching.h"

PYBIND11_MODULE(mast3r_slam_cpu_backends, m) {
    m.doc() = "CPU backend for MASt3R-SLAM (OpenMP + SIMD)";

    // Matching kernels
    m.def("iter_proj", &cpu_backend::iter_proj,
          "Iterative projection matching (CPU)",
          py::arg("rays_img_with_grad"),
          py::arg("pts_3d_norm"),
          py::arg("p_init"),
          py::arg("max_iter"),
          py::arg("lambda_init"),
          py::arg("cost_thresh"));

    m.def("refine_matches", &cpu_backend::refine_matches,
          "Refine matches using descriptor correlation (CPU)",
          py::arg("D11"),
          py::arg("D21"),
          py::arg("p1"),
          py::arg("radius"),
          py::arg("dilation_max"));

    // Gauss-Newton kernels
    m.def("gauss_newton_rays", &cpu_backend::gauss_newton_rays,
          "Gauss-Newton optimization for ray alignment (CPU)",
          py::arg("poses"),
          py::arg("points"),
          py::arg("confidences"),
          py::arg("ii"),
          py::arg("jj"),
          py::arg("idx_ii2jj"),
          py::arg("valid_match"),
          py::arg("Q"),
          py::arg("sigma_ray"),
          py::arg("sigma_dist"),
          py::arg("C_thresh"),
          py::arg("Q_thresh"),
          py::arg("max_iter"),
          py::arg("delta_thresh"));

    m.def("gauss_newton_calib", &cpu_backend::gauss_newton_calib,
          "Gauss-Newton optimization with calibration (CPU)",
          py::arg("poses"),
          py::arg("points"),
          py::arg("confidences"),
          py::arg("K"),
          py::arg("ii"),
          py::arg("jj"),
          py::arg("idx_ii2jj"),
          py::arg("valid_match"),
          py::arg("Q"),
          py::arg("height"),
          py::arg("width"),
          py::arg("pixel_border"),
          py::arg("z_eps"),
          py::arg("sigma_pixel"),
          py::arg("sigma_depth"),
          py::arg("C_thresh"),
          py::arg("Q_thresh"),
          py::arg("max_iter"),
          py::arg("delta_thresh"));
}
