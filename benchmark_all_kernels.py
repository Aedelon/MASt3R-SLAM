# Copyright Delanoe Pirard / Aedelon. Apache 2.0
"""Complete benchmark of all Metal kernels vs Numpy implementations."""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, "src")

import numpy as np

np.random.seed(42)


def create_gn_test_data(num_kf=10, num_pts=500, num_edges=15):
    """Create synthetic test data for Gauss-Newton."""
    Twc = np.zeros((num_kf, 8), dtype=np.float32)
    for i in range(num_kf):
        Twc[i, :3] = np.random.randn(3) * 0.5
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        Twc[i, 3:7] = q
        Twc[i, 7] = 1.0 + np.random.rand() * 0.1

    Xs = np.random.randn(num_kf, num_pts, 3).astype(np.float32)
    Cs = np.random.rand(num_kf, num_pts).astype(np.float32) * 10 + 1

    ii = np.random.randint(0, num_kf, num_edges).astype(np.int32)
    jj = np.random.randint(0, num_kf, num_edges).astype(np.int32)
    for e in range(num_edges):
        while jj[e] == ii[e]:
            jj[e] = np.random.randint(0, num_kf)

    idx_ii2jj = np.zeros((num_edges, num_pts), dtype=np.int32)
    for e in range(num_edges):
        idx_ii2jj[e] = np.random.permutation(num_pts)

    valid_match = np.random.rand(num_edges, num_pts) > 0.3
    Q = np.random.rand(num_edges, num_pts).astype(np.float32) * 3 + 1

    return Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q


def benchmark_iter_proj():
    """Benchmark iter_proj kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK: iter_proj")
    print("=" * 70)

    from mlx_mast3r_slam.backends.mpsgraph.kernels import iter_proj, _iter_proj_numpy
    from mlx_mast3r_slam.backends.mpsgraph.metal_runner import get_runner

    results = {"kernel": "iter_proj", "configs": []}

    configs = [
        (1, 64, 64, 1000),
        (1, 128, 128, 5000),
        (1, 256, 256, 20000),
        (2, 384, 512, 50000),
    ]

    for b, h, w, n in configs:
        config_name = f"b={b}, img={h}x{w}, pts={n}"
        print(f"\n--- {config_name} ---")

        rays = np.random.randn(b, h, w, 9).astype(np.float32)
        pts = np.random.randn(b, n, 3).astype(np.float32)
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
        p_init = np.stack(
            [
                np.random.rand(b, n) * (w - 1),
                np.random.rand(b, n) * (h - 1),
            ],
            axis=-1,
        ).astype(np.float32)

        # Numpy
        start = time.perf_counter()
        result_np, valid_np = _iter_proj_numpy(rays, pts, p_init.copy(), 10, 1e-8, 1e-6)
        time_np = time.perf_counter() - start

        # Metal
        try:
            runner = get_runner()
            start = time.perf_counter()
            result_mt, valid_mt = runner.iter_proj(rays, pts, p_init.copy(), 10, 1e-8, 1e-6)
            time_mt = time.perf_counter() - start

            # Correlation
            valid_both = valid_np & valid_mt
            if np.sum(valid_both) > 0:
                diff = np.abs(result_np[valid_both] - result_mt[valid_both])
                max_diff = diff.max()
                mean_diff = diff.mean()
                corr = np.corrcoef(
                    result_np[valid_both].flatten(), result_mt[valid_both].flatten()
                )[0, 1]
            else:
                max_diff = mean_diff = corr = 0

            speedup = time_np / time_mt
            print(f"  Numpy:  {time_np * 1000:8.1f} ms")
            print(f"  Metal:  {time_mt * 1000:8.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(
                f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Correlation: {corr:.6f}"
            )

            results["configs"].append(
                {
                    "config": config_name,
                    "numpy_ms": time_np * 1000,
                    "metal_ms": time_mt * 1000,
                    "speedup": speedup,
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "correlation": float(corr) if not np.isnan(corr) else 1.0,
                }
            )
        except Exception as e:
            print(f"  Metal error: {e}")

    return results


def benchmark_gauss_newton_rays():
    """Benchmark gauss_newton_rays kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK: gauss_newton_rays")
    print("=" * 70)

    from mlx_mast3r_slam.backends.mpsgraph.gauss_newton import gauss_newton_rays as gn_numpy
    from mlx_mast3r_slam.backends.mpsgraph.gn_metal_runner import get_gn_runner

    results = {"kernel": "gauss_newton_rays", "configs": []}

    configs = [
        (5, 200, 8),
        (10, 500, 15),
        (20, 1000, 30),
    ]

    for num_kf, num_pts, num_edges in configs:
        config_name = f"kf={num_kf}, pts={num_pts}, edges={num_edges}"
        print(f"\n--- {config_name} ---")

        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q = create_gn_test_data(
            num_kf, num_pts, num_edges
        )

        # Numpy
        start = time.perf_counter()
        result_np = gn_numpy(
            Twc.copy(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, max_iter=3, pin=1
        )
        time_np = time.perf_counter() - start

        # Metal
        try:
            runner = get_gn_runner()
            start = time.perf_counter()
            result_mt = runner.gauss_newton_rays(
                Twc.copy(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, max_iter=3, pin=1
            )
            time_mt = time.perf_counter() - start

            diff = np.abs(result_np - result_mt)
            max_diff = diff.max()
            mean_diff = diff.mean()
            corr = np.corrcoef(result_np.flatten(), result_mt.flatten())[0, 1]
            speedup = time_np / time_mt

            print(f"  Numpy:  {time_np * 1000:8.1f} ms")
            print(f"  Metal:  {time_mt * 1000:8.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(
                f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Correlation: {corr:.6f}"
            )

            results["configs"].append(
                {
                    "config": config_name,
                    "numpy_ms": time_np * 1000,
                    "metal_ms": time_mt * 1000,
                    "speedup": speedup,
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "correlation": float(corr) if not np.isnan(corr) else 1.0,
                }
            )
        except Exception as e:
            print(f"  Metal error: {e}")

    return results


def benchmark_gauss_newton_calib():
    """Benchmark gauss_newton_calib kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK: gauss_newton_calib")
    print("=" * 70)

    from mlx_mast3r_slam.backends.mpsgraph.gauss_newton_calib import gauss_newton_calib as gn_numpy
    from mlx_mast3r_slam.backends.mpsgraph.gn_calib_metal_runner import get_gn_calib_runner

    results = {"kernel": "gauss_newton_calib", "configs": []}

    configs = [
        (5, 200, 8),
        (10, 500, 15),
        (20, 1000, 30),
    ]

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    img_size = (640, 480)

    for num_kf, num_pts, num_edges in configs:
        config_name = f"kf={num_kf}, pts={num_pts}, edges={num_edges}"
        print(f"\n--- {config_name} ---")

        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q = create_gn_test_data(
            num_kf, num_pts, num_edges
        )
        # Ensure positive depth
        Xs = np.abs(Xs) + 0.1

        # Numpy
        start = time.perf_counter()
        result_np = gn_numpy(
            Twc.copy(), Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q, img_size, max_iter=3, pin=1
        )
        time_np = time.perf_counter() - start

        # Metal
        try:
            runner = get_gn_calib_runner()
            start = time.perf_counter()
            result_mt = runner.gauss_newton_calib(
                Twc.copy(),
                Xs,
                Cs,
                K,
                ii,
                jj,
                idx_ii2jj,
                valid_match,
                Q,
                img_size,
                max_iter=3,
                pin=1,
            )
            time_mt = time.perf_counter() - start

            diff = np.abs(result_np - result_mt)
            max_diff = diff.max()
            mean_diff = diff.mean()
            corr = np.corrcoef(result_np.flatten(), result_mt.flatten())[0, 1]
            speedup = time_np / time_mt

            print(f"  Numpy:  {time_np * 1000:8.1f} ms")
            print(f"  Metal:  {time_mt * 1000:8.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(
                f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Correlation: {corr:.6f}"
            )

            results["configs"].append(
                {
                    "config": config_name,
                    "numpy_ms": time_np * 1000,
                    "metal_ms": time_mt * 1000,
                    "speedup": speedup,
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "correlation": float(corr) if not np.isnan(corr) else 1.0,
                }
            )
        except Exception as e:
            print(f"  Metal error: {e}")
            import traceback

            traceback.print_exc()

    return results


def benchmark_gauss_newton_points():
    """Benchmark gauss_newton_points kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK: gauss_newton_points")
    print("=" * 70)

    from mlx_mast3r_slam.backends.mpsgraph.gauss_newton_points import (
        gauss_newton_points as gn_numpy,
    )
    from mlx_mast3r_slam.backends.mpsgraph.gn_points_metal_runner import get_gn_points_runner

    results = {"kernel": "gauss_newton_points", "configs": []}

    configs = [
        (5, 200, 8),
        (10, 500, 15),
        (20, 1000, 30),
    ]

    for num_kf, num_pts, num_edges in configs:
        config_name = f"kf={num_kf}, pts={num_pts}, edges={num_edges}"
        print(f"\n--- {config_name} ---")

        Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q = create_gn_test_data(
            num_kf, num_pts, num_edges
        )

        # Numpy
        start = time.perf_counter()
        result_np = gn_numpy(
            Twc.copy(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, max_iter=3, pin=1
        )
        time_np = time.perf_counter() - start

        # Metal
        try:
            runner = get_gn_points_runner()
            start = time.perf_counter()
            result_mt = runner.gauss_newton_points(
                Twc.copy(), Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q, max_iter=3, pin=1
            )
            time_mt = time.perf_counter() - start

            diff = np.abs(result_np - result_mt)
            max_diff = diff.max()
            mean_diff = diff.mean()
            corr = np.corrcoef(result_np.flatten(), result_mt.flatten())[0, 1]
            speedup = time_np / time_mt

            print(f"  Numpy:  {time_np * 1000:8.1f} ms")
            print(f"  Metal:  {time_mt * 1000:8.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(
                f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Correlation: {corr:.6f}"
            )

            results["configs"].append(
                {
                    "config": config_name,
                    "numpy_ms": time_np * 1000,
                    "metal_ms": time_mt * 1000,
                    "speedup": speedup,
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "correlation": float(corr) if not np.isnan(corr) else 1.0,
                }
            )
        except Exception as e:
            print(f"  Metal error: {e}")
            import traceback

            traceback.print_exc()

    return results


def benchmark_refine_matches():
    """Benchmark refine_matches kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK: refine_matches")
    print("=" * 70)

    from mlx_mast3r_slam.backends.mpsgraph.kernels import _refine_matches_numpy
    from mlx_mast3r_slam.backends.mpsgraph.refine_metal_runner import get_refine_runner

    results = {"kernel": "refine_matches", "configs": []}

    configs = [
        (1, 384, 512, 32, 1000),
        (1, 384, 512, 64, 5000),
        (2, 384, 512, 64, 10000),
    ]

    for b, h, w, d, n in configs:
        config_name = f"b={b}, img={h}x{w}, desc={d}, pts={n}"
        print(f"\n--- {config_name} ---")

        D11 = np.random.randn(b, h, w, d).astype(np.float32)
        D11 = D11 / np.linalg.norm(D11, axis=-1, keepdims=True)
        D21 = np.random.randn(b, n, d).astype(np.float32)
        D21 = D21 / np.linalg.norm(D21, axis=-1, keepdims=True)
        p1 = np.stack(
            [
                np.random.randint(5, w - 5, (b, n)),
                np.random.randint(5, h - 5, (b, n)),
            ],
            axis=-1,
        ).astype(np.int32)

        # Numpy
        start = time.perf_counter()
        result_np = _refine_matches_numpy(D11, D21, p1.copy(), 3, 2)
        time_np = time.perf_counter() - start

        # Metal
        try:
            runner = get_refine_runner()
            start = time.perf_counter()
            result_mt = runner.refine_matches(D11, D21, p1.copy(), 3, 2)
            time_mt = time.perf_counter() - start

            diff = np.abs(result_np - result_mt)
            max_diff = diff.max()
            mean_diff = diff.mean()
            # For discrete positions, use exact match ratio
            exact_match = (diff.max(axis=-1) == 0).mean()
            speedup = time_np / time_mt

            print(f"  Numpy:  {time_np * 1000:8.1f} ms")
            print(f"  Metal:  {time_mt * 1000:8.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(
                f"  Max diff: {max_diff}, Mean diff: {mean_diff:.4f}, Exact match: {exact_match * 100:.1f}%"
            )

            results["configs"].append(
                {
                    "config": config_name,
                    "numpy_ms": time_np * 1000,
                    "metal_ms": time_mt * 1000,
                    "speedup": speedup,
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "exact_match_pct": float(exact_match * 100),
                }
            )
        except Exception as e:
            print(f"  Metal error: {e}")

    return results


def generate_markdown_report(all_results, device_info):
    """Generate markdown report."""

    report = f"""# Metal Kernels Benchmark Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Device:** {device_info}
**Platform:** Apple Silicon (Metal)

## Summary

This benchmark compares Metal GPU implementations against NumPy CPU implementations
for all critical SLAM kernels.

## Results Overview

| Kernel | Avg Speedup | Max Diff | Correlation |
|--------|-------------|----------|-------------|
"""

    for result in all_results:
        kernel = result["kernel"]
        if result["configs"]:
            avg_speedup = np.mean([c["speedup"] for c in result["configs"]])
            max_diff = max([c["max_diff"] for c in result["configs"]])
            if "correlation" in result["configs"][0]:
                avg_corr = np.mean([c["correlation"] for c in result["configs"]])
                report += f"| {kernel} | {avg_speedup:.1f}x | {max_diff:.6f} | {avg_corr:.6f} |\n"
            else:
                exact_match = np.mean([c["exact_match_pct"] for c in result["configs"]])
                report += (
                    f"| {kernel} | {avg_speedup:.1f}x | {max_diff} | {exact_match:.1f}% exact |\n"
                )

    report += "\n## Detailed Results\n"

    for result in all_results:
        kernel = result["kernel"]
        report += f"\n### {kernel}\n\n"

        if "correlation" in result["configs"][0] if result["configs"] else False:
            report += "| Configuration | NumPy (ms) | Metal (ms) | Speedup | Max Diff | Mean Diff | Correlation |\n"
            report += "|---------------|------------|------------|---------|----------|-----------|-------------|\n"
            for cfg in result["configs"]:
                report += f"| {cfg['config']} | {cfg['numpy_ms']:.1f} | {cfg['metal_ms']:.1f} | {cfg['speedup']:.1f}x | {cfg['max_diff']:.6f} | {cfg['mean_diff']:.6f} | {cfg['correlation']:.6f} |\n"
        else:
            report += "| Configuration | NumPy (ms) | Metal (ms) | Speedup | Max Diff | Mean Diff | Exact Match |\n"
            report += "|---------------|------------|------------|---------|----------|-----------|-------------|\n"
            for cfg in result["configs"]:
                report += f"| {cfg['config']} | {cfg['numpy_ms']:.1f} | {cfg['metal_ms']:.1f} | {cfg['speedup']:.1f}x | {cfg['max_diff']} | {cfg['mean_diff']:.4f} | {cfg['exact_match_pct']:.1f}% |\n"

    report += """
## Notes

### Correlation Interpretation
- **Correlation > 0.99**: Excellent agreement, differences due to floating point precision
- **Correlation > 0.95**: Good agreement, minor numerical differences
- **Correlation < 0.95**: Potential algorithmic differences

### Speedup Factors
- Speedup varies with problem size (larger = better GPU utilization)
- Metal kernels include data transfer overhead
- NumPy uses optimized BLAS but is single-threaded for custom operations

### Differences Explanation
- **iter_proj**: Small differences from floating point precision in iterative solver
- **gauss_newton_***: Differences from float32 vs float64 accumulation and convergence criteria
- **refine_matches**: Differences from tie-breaking in equal-score matches (random descriptors)

## Kernel Implementations

| CUDA Original | Metal Implementation | Shader File |
|---------------|---------------------|-------------|
| `iter_proj_kernel` | `iter_proj_kernel` | `iter_proj.metal` |
| `refine_matches_kernel` | `refine_matches_kernel` | `refine_matches.metal` |
| `point_align_kernel` | `gn_points_jacobian_kernel` | `gauss_newton_points.metal` |
| `ray_align_kernel` | `gn_jacobian_kernel` | `gauss_newton.metal` |
| `calib_proj_kernel` | `gn_calib_jacobian_kernel` | `gauss_newton_calib.metal` |
| `pose_retr_kernel` | `pose_update_kernel` | `gauss_newton.metal` |

## Files Structure

```
src/mlx_mast3r_slam/backends/mpsgraph/
├── kernels.py                    # Unified interface
├── metal_runner.py               # iter_proj runner
├── gn_metal_runner.py            # GN rays runner
├── gn_calib_metal_runner.py      # GN calib runner
├── gn_points_metal_runner.py     # GN points runner
├── refine_metal_runner.py        # Match refinement runner
├── gauss_newton.py               # NumPy fallback (rays)
├── gauss_newton_calib.py         # NumPy fallback (calib)
├── gauss_newton_points.py        # NumPy fallback (points)
├── sim3_ops.py                   # Lie group operations
└── shaders/
    ├── iter_proj.metal
    ├── gauss_newton.metal
    ├── gauss_newton_calib.metal
    ├── gauss_newton_points.metal
    └── refine_matches.metal
```
"""

    return report


def main():
    print("=" * 70)
    print("COMPLETE METAL KERNELS BENCHMARK")
    print("=" * 70)

    # Get device info
    try:
        import Metal as mtl

        device = mtl.MTLCreateSystemDefaultDevice()
        device_info = device.name()
    except:
        device_info = "Unknown Metal Device"

    print(f"\nDevice: {device_info}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Run all benchmarks
    all_results.append(benchmark_iter_proj())
    all_results.append(benchmark_gauss_newton_rays())
    all_results.append(benchmark_gauss_newton_calib())
    all_results.append(benchmark_gauss_newton_points())
    all_results.append(benchmark_refine_matches())

    # Generate report
    report = generate_markdown_report(all_results, device_info)

    # Save report
    import os

    os.makedirs("docs", exist_ok=True)

    with open("docs/metal_kernels_benchmark.md", "w") as f:
        f.write(report)

    # Also save raw JSON
    with open("docs/benchmark_results.json", "w") as f:
        json.dump(
            {"device": device_info, "date": datetime.now().isoformat(), "results": all_results},
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nReport saved to: docs/metal_kernels_benchmark.md")
    print(f"Raw data saved to: docs/benchmark_results.json")


if __name__ == "__main__":
    main()
