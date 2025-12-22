#!/usr/bin/env python3
# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Comprehensive benchmark for MASt3R-SLAM backends.

Compares CPU (OpenMP) vs Metal (Apple Silicon GPU) performance.
"""

import time
from dataclasses import dataclass
from typing import Callable

import torch

# Benchmark configuration
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


@dataclass
class BenchmarkResult:
    name: str
    size: str
    cpu_ms: float
    metal_ms: float
    speedup: float


def time_fn(
    fn: Callable, warmup: int = WARMUP_RUNS, runs: int = BENCHMARK_RUNS
) -> float:
    """Time a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Sync before timing
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        fn()

    # Sync after
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000  # ms


def create_test_data(
    batch_size: int,
    n_points: int,
    H: int,
    W: int,
    fdim: int,
    device: torch.device,
):
    """Create test data for matching benchmarks."""
    rays = torch.randn(batch_size, H, W, 6, device=device, dtype=torch.float32)
    pts_3d = torch.randn(batch_size, n_points, 3, device=device, dtype=torch.float32)
    p_init = torch.rand(batch_size, n_points, 2, device=device, dtype=torch.float32)
    p_init[..., 0] *= W - 1
    p_init[..., 1] *= H - 1

    D11 = torch.randn(batch_size, H, W, fdim, device=device, dtype=torch.float32)
    # D21 contains descriptors at query positions: [batch, n_points, fdim]
    D21 = torch.randn(batch_size, n_points, fdim, device=device, dtype=torch.float32)
    # p1 has shape [batch_size, n_points, 2] with pixel coordinates
    p1 = torch.zeros(batch_size, n_points, 2, device=device, dtype=torch.int64)
    p1[..., 0] = torch.randint(0, W, (batch_size, n_points), device=device)  # x
    p1[..., 1] = torch.randint(0, H, (batch_size, n_points), device=device)  # y

    return rays, pts_3d, p_init, D11, D21, p1


def create_gn_test_data(
    num_poses: int,
    num_points: int,
    num_edges: int,
    H: int,
    W: int,
    device: torch.device,
):
    """Create test data for Gauss-Newton benchmarks."""
    # Poses: [num_poses, 8] = [tx, ty, tz, qx, qy, qz, qw, s]
    poses = torch.zeros(num_poses, 8, device=device, dtype=torch.float32)
    poses[:, 6] = 1.0  # qw = 1 (identity rotation)
    poses[:, 7] = 1.0  # s = 1 (identity scale)
    # Add some translation variation
    poses[:, :3] = torch.randn(num_poses, 3, device=device) * 0.1

    # Points: [num_poses, num_points, 3]
    points = torch.randn(num_poses, num_points, 3, device=device, dtype=torch.float32)
    points[..., 2] = points[..., 2].abs() + 0.5  # Positive depth

    # Confidences: [num_poses, num_points, 1]
    confidences = torch.rand(
        num_poses, num_points, 1, device=device, dtype=torch.float32
    )

    # Edges
    ii = torch.randint(0, num_poses, (num_edges,), device=device, dtype=torch.int64)
    jj = torch.randint(0, num_poses, (num_edges,), device=device, dtype=torch.int64)
    # Ensure ii != jj
    jj = (
        ii + 1 + torch.randint(0, num_poses - 1, (num_edges,), device=device)
    ) % num_poses

    # Match indices
    idx_ii2jj = torch.randint(
        0, num_points, (num_edges, num_points), device=device, dtype=torch.int64
    )
    valid_match = torch.rand(num_edges, num_points, 1, device=device) > 0.2
    Q = torch.rand(num_edges, num_points, 1, device=device, dtype=torch.float32)

    # Camera intrinsics
    K = torch.tensor(
        [
            [500.0, 0.0, W / 2],
            [0.0, 500.0, H / 2],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    return poses, points, confidences, ii, jj, idx_ii2jj, valid_match, Q, K


def benchmark_iter_proj(
    cpu_backend, metal_backend, sizes: list[tuple]
) -> list[BenchmarkResult]:
    """Benchmark iter_proj kernel."""
    results = []

    for batch_size, n_points, H, W in sizes:
        size_str = f"{batch_size}x{n_points}pts {H}x{W}"
        print(f"  iter_proj {size_str}...", end=" ", flush=True)

        # CPU
        rays, pts_3d, p_init, _, _, _ = create_test_data(
            batch_size, n_points, H, W, 24, torch.device("cpu")
        )
        cpu_ms = time_fn(
            lambda: cpu_backend.iter_proj(rays, pts_3d, p_init, 10, 0.1, 1e-4)
        )

        # Metal
        rays_mps = rays.to("mps")
        pts_3d_mps = pts_3d.to("mps")
        p_init_mps = p_init.to("mps")
        metal_ms = time_fn(
            lambda: metal_backend.iter_proj(
                rays_mps, pts_3d_mps, p_init_mps, 10, 0.1, 1e-4
            )
        )

        speedup = cpu_ms / metal_ms if metal_ms > 0 else 0
        results.append(
            BenchmarkResult("iter_proj", size_str, cpu_ms, metal_ms, speedup)
        )
        print(f"CPU={cpu_ms:.2f}ms Metal={metal_ms:.2f}ms ({speedup:.1f}x)")

    return results


def benchmark_refine_matches(
    cpu_backend, metal_backend, sizes: list[tuple]
) -> list[BenchmarkResult]:
    """Benchmark refine_matches kernel."""
    results = []

    for batch_size, n_points, H, W, fdim in sizes:
        size_str = f"{batch_size}x{n_points}pts {H}x{W}x{fdim}"
        print(f"  refine_matches {size_str}...", end=" ", flush=True)

        # CPU
        _, _, _, D11, D21, p1 = create_test_data(
            batch_size, n_points, H, W, fdim, torch.device("cpu")
        )
        cpu_ms = time_fn(lambda: cpu_backend.refine_matches(D11, D21, p1, 4, 4))

        # Metal
        D11_mps = D11.to("mps")
        D21_mps = D21.to("mps")
        p1_mps = p1.to("mps")
        metal_ms = time_fn(
            lambda: metal_backend.refine_matches(D11_mps, D21_mps, p1_mps, 4, 4)
        )

        speedup = cpu_ms / metal_ms if metal_ms > 0 else 0
        results.append(
            BenchmarkResult("refine_matches", size_str, cpu_ms, metal_ms, speedup)
        )
        print(f"CPU={cpu_ms:.2f}ms Metal={metal_ms:.2f}ms ({speedup:.1f}x)")

    return results


def benchmark_gauss_newton_rays(
    cpu_backend, metal_backend, sizes: list[tuple]
) -> list[BenchmarkResult]:
    """Benchmark gauss_newton_rays kernel."""
    results = []

    for num_poses, num_points, num_edges in sizes:
        size_str = f"{num_poses}poses {num_points}pts {num_edges}edges"
        print(f"  gn_rays {size_str}...", end=" ", flush=True)

        H, W = 384, 512

        # CPU
        poses, points, conf, ii, jj, idx, valid, Q, _ = create_gn_test_data(
            num_poses, num_points, num_edges, H, W, torch.device("cpu")
        )

        def run_cpu():
            poses_copy = poses.clone()
            return cpu_backend.gauss_newton_rays(
                poses_copy,
                points,
                conf,
                ii,
                jj,
                idx,
                valid,
                Q,
                0.01,
                0.1,
                0.5,
                0.5,
                3,
                1e-4,
            )

        cpu_ms = time_fn(run_cpu)

        # Metal
        poses_mps = poses.to("mps")
        points_mps = points.to("mps")
        conf_mps = conf.to("mps")
        ii_mps = ii.to("mps")
        jj_mps = jj.to("mps")
        idx_mps = idx.to("mps")
        valid_mps = valid.to("mps")
        Q_mps = Q.to("mps")

        def run_metal():
            poses_copy = poses_mps.clone()
            return metal_backend.gauss_newton_rays(
                poses_copy,
                points_mps,
                conf_mps,
                ii_mps,
                jj_mps,
                idx_mps,
                valid_mps,
                Q_mps,
                0.01,
                0.1,
                0.5,
                0.5,
                3,
                1e-4,
            )

        metal_ms = time_fn(run_metal)

        speedup = cpu_ms / metal_ms if metal_ms > 0 else 0
        results.append(BenchmarkResult("gn_rays", size_str, cpu_ms, metal_ms, speedup))
        print(f"CPU={cpu_ms:.2f}ms Metal={metal_ms:.2f}ms ({speedup:.1f}x)")

    return results


def benchmark_gauss_newton_calib(
    cpu_backend, metal_backend, sizes: list[tuple]
) -> list[BenchmarkResult]:
    """Benchmark gauss_newton_calib kernel."""
    results = []

    for num_poses, num_points, num_edges in sizes:
        size_str = f"{num_poses}poses {num_points}pts {num_edges}edges"
        print(f"  gn_calib {size_str}...", end=" ", flush=True)

        H, W = 384, 512

        # CPU
        poses, points, conf, ii, jj, idx, valid, Q, K = create_gn_test_data(
            num_poses, num_points, num_edges, H, W, torch.device("cpu")
        )

        def run_cpu():
            poses_copy = poses.clone()
            return cpu_backend.gauss_newton_calib(
                poses_copy,
                points,
                conf,
                K,
                ii,
                jj,
                idx,
                valid,
                Q,
                H,
                W,
                4,
                0.01,
                1.0,
                0.1,
                0.5,
                0.5,
                3,
                1e-4,
            )

        cpu_ms = time_fn(run_cpu)

        # Metal
        poses_mps = poses.to("mps")
        points_mps = points.to("mps")
        conf_mps = conf.to("mps")
        K_mps = K.to("mps")
        ii_mps = ii.to("mps")
        jj_mps = jj.to("mps")
        idx_mps = idx.to("mps")
        valid_mps = valid.to("mps")
        Q_mps = Q.to("mps")

        def run_metal():
            poses_copy = poses_mps.clone()
            return metal_backend.gauss_newton_calib(
                poses_copy,
                points_mps,
                conf_mps,
                K_mps,
                ii_mps,
                jj_mps,
                idx_mps,
                valid_mps,
                Q_mps,
                H,
                W,
                4,
                0.01,
                1.0,
                0.1,
                0.5,
                0.5,
                3,
                1e-4,
            )

        metal_ms = time_fn(run_metal)

        speedup = cpu_ms / metal_ms if metal_ms > 0 else 0
        results.append(BenchmarkResult("gn_calib", size_str, cpu_ms, metal_ms, speedup))
        print(f"CPU={cpu_ms:.2f}ms Metal={metal_ms:.2f}ms ({speedup:.1f}x)")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Kernel':<20} {'Size':<30} {'CPU (ms)':<12} {'Metal (ms)':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for r in results:
        speedup_str = f"{r.speedup:.2f}x"
        if r.speedup > 1:
            speedup_str = f"\033[92m{speedup_str}\033[0m"  # Green
        elif r.speedup < 1:
            speedup_str = f"\033[91m{speedup_str}\033[0m"  # Red

        print(
            f"{r.name:<20} {r.size:<30} {r.cpu_ms:<12.2f} {r.metal_ms:<12.2f} {speedup_str}"
        )

    print("-" * 80)

    # Average speedup by kernel
    kernels = set(r.name for r in results)
    for kernel in sorted(kernels):
        kernel_results = [r for r in results if r.name == kernel]
        avg_speedup = sum(r.speedup for r in kernel_results) / len(kernel_results)
        print(f"{kernel:<20} {'Average':<30} {'':<12} {'':<12} {avg_speedup:.2f}x")

    # Overall average
    avg_all = sum(r.speedup for r in results) / len(results)
    print(f"{'OVERALL':<20} {'Average':<30} {'':<12} {'':<12} {avg_all:.2f}x")
    print("=" * 80)


def main():
    print("=" * 80)
    print("MASt3R-SLAM Backend Benchmark")
    print("=" * 80)
    print(
        f"Device: {torch.backends.mps.is_available() and 'Apple Silicon (MPS)' or 'CPU only'}"
    )
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup runs: {WARMUP_RUNS}, Benchmark runs: {BENCHMARK_RUNS}")
    print()

    # Load backends
    print("Loading backends...")
    try:
        from mast3r_slam.backends.cpu import CPUBackend

        cpu_backend = CPUBackend()
        print("  ✓ CPU backend loaded")
    except Exception as e:
        print(f"  ✗ CPU backend failed: {e}")
        return

    try:
        from mast3r_slam.backends.metal import MetalBackend

        metal_backend = MetalBackend()
        print("  ✓ Metal backend loaded")
    except Exception as e:
        print(f"  ✗ Metal backend failed: {e}")
        return

    print()

    all_results = []

    # iter_proj benchmarks
    print("iter_proj benchmarks:")
    iter_proj_sizes = [
        (1, 1000, 384, 512),
        (1, 5000, 384, 512),
        (1, 10000, 384, 512),
        (1, 20000, 384, 512),
    ]
    all_results.extend(benchmark_iter_proj(cpu_backend, metal_backend, iter_proj_sizes))
    print()

    # refine_matches benchmarks
    print("refine_matches benchmarks:")
    refine_sizes = [
        (1, 1000, 384, 512, 24),
        (1, 5000, 384, 512, 24),
        (1, 10000, 384, 512, 24),
        (1, 20000, 384, 512, 24),
    ]
    all_results.extend(
        benchmark_refine_matches(cpu_backend, metal_backend, refine_sizes)
    )
    print()

    # gauss_newton_rays benchmarks
    print("gauss_newton_rays benchmarks:")
    gn_sizes = [
        (10, 1000, 20),
        (20, 2000, 50),
        (50, 5000, 100),
        (100, 10000, 200),
    ]
    all_results.extend(
        benchmark_gauss_newton_rays(cpu_backend, metal_backend, gn_sizes)
    )
    print()

    # gauss_newton_calib benchmarks
    print("gauss_newton_calib benchmarks:")
    all_results.extend(
        benchmark_gauss_newton_calib(cpu_backend, metal_backend, gn_sizes)
    )
    print()

    # Summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
