// Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
/**
 * Python bindings for Metal backend.
 */

#include "metal_ops.h"

PYBIND11_MODULE(mast3r_slam_metal_backends, m) {
    m.doc() = "Metal backend for MASt3R-SLAM (Apple Silicon GPU)";

    m.def("initialize", &metal_backend::initialize,
          "Initialize Metal device and compile shaders");

    m.def("is_available", &metal_backend::is_available,
          "Check if Metal backend is available");

    // Matching kernels
    m.def("iter_proj", &metal_backend::iter_proj,
          "Iterative projection matching (Metal)",
          py::arg("rays_img_with_grad"),
          py::arg("pts_3d_norm"),
          py::arg("p_init"),
          py::arg("max_iter"),
          py::arg("lambda_init"),
          py::arg("cost_thresh"));

    m.def("refine_matches", &metal_backend::refine_matches,
          "Refine matches using descriptor correlation (Metal)",
          py::arg("D11"),
          py::arg("D21"),
          py::arg("p1"),
          py::arg("radius"),
          py::arg("dilation_max"));

    // Gauss-Newton kernels
    m.def("gauss_newton_rays", &metal_backend::gauss_newton_rays,
          "Gauss-Newton optimization for ray alignment (Metal)",
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

    m.def("gauss_newton_calib", &metal_backend::gauss_newton_calib,
          "Gauss-Newton optimization with calibration (Metal)",
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
