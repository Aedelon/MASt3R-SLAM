# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Comprehensive benchmark for all backends.
Finds the crossover point where Metal becomes faster than CPU.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backends
cpu_path = Path(__file__).parent.parent / "mast3r_slam" / "backends" / "cpu"
metal_path = Path(__file__).parent.parent / "mast3r_slam" / "backends" / "metal"
sys.path.insert(0, str(cpu_path))
sys.path.insert(0, str(metal_path))

import mast3r_slam_cpu_backends
import mast3r_slam_metal_backends


def benchmark_refine_matches(n_runs: int = 10):
    """Benchmark refine_matches across different scales."""
    print("=" * 70)
    print("BENCHMARK: refine_matches - Finding CPU/Metal crossover")
    print("=" * 70)

    configs = [
        {"n_points": 1000, "H": 128, "W": 128, "fdim": 24},
        {"n_points": 5000, "H": 256, "W": 256, "fdim": 24},
        {"n_points": 10000, "H": 256, "W": 256, "fdim": 24},
        {"n_points": 20000, "H": 256, "W": 256, "fdim": 24},
        {"n_points": 50000, "H": 512, "W": 512, "fdim": 24},
        {"n_points": 100000, "H": 512, "W": 512, "fdim": 24},
    ]

    print(
        f"\n{'Points':>10} {'Image':>10} {'CPU (ms)':>12} {'Metal (ms)':>12} {'Speedup':>10}"
    )
    print("-" * 60)

    for cfg in configs:
        n_points = cfg["n_points"]
        H, W = cfg["H"], cfg["W"]
        fdim = cfg["fdim"]

        # Create test data
        D11 = torch.nn.functional.normalize(
            torch.randn(1, H, W, fdim, dtype=torch.float32), dim=-1
        )
        D21 = torch.nn.functional.normalize(
            torch.randn(1, n_points, fdim, dtype=torch.float32), dim=-1
        )
        p1 = torch.zeros(1, n_points, 2, dtype=torch.int64)
        p1[..., 0] = torch.randint(5, W - 5, (1, n_points))
        p1[..., 1] = torch.randint(5, H - 5, (1, n_points))

        # Warmup CPU
        for _ in range(2):
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )

        # Benchmark CPU
        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = sum(cpu_times) / len(cpu_times) * 1000

        # Warmup Metal
        for _ in range(2):
            mast3r_slam_metal_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )

        # Benchmark Metal
        metal_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_metal_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )
            metal_times.append(time.perf_counter() - start)
        metal_avg = sum(metal_times) / len(metal_times) * 1000

        speedup = cpu_avg / metal_avg
        winner = "Metal" if speedup > 1 else "CPU"

        print(
            f"{n_points:>10} {H}x{W:>7} {cpu_avg:>12.2f} {metal_avg:>12.2f} {speedup:>9.2f}x ({winner})"
        )


def benchmark_iter_proj(n_runs: int = 10):
    """Benchmark iter_proj across different scales."""
    print("\n" + "=" * 70)
    print("BENCHMARK: iter_proj - Finding CPU/Metal crossover")
    print("=" * 70)

    configs = [
        {"n_points": 1000, "H": 128, "W": 128},
        {"n_points": 5000, "H": 256, "W": 256},
        {"n_points": 10000, "H": 256, "W": 256},
        {"n_points": 20000, "H": 256, "W": 256},
        {"n_points": 50000, "H": 512, "W": 512},
    ]

    print(
        f"\n{'Points':>10} {'Image':>10} {'CPU (ms)':>12} {'Metal (ms)':>12} {'Speedup':>10}"
    )
    print("-" * 60)

    for cfg in configs:
        n_points = cfg["n_points"]
        H, W = cfg["H"], cfg["W"]

        rays_img = torch.randn(1, H, W, 9, dtype=torch.float32)
        rays_img[..., :3] = torch.nn.functional.normalize(rays_img[..., :3], dim=-1)

        pts_3d = torch.nn.functional.normalize(
            torch.randn(1, n_points, 3, dtype=torch.float32), dim=-1
        )

        p_init = torch.rand(1, n_points, 2, dtype=torch.float32)
        p_init[..., 0] = p_init[..., 0] * (W - 4) + 2
        p_init[..., 1] = p_init[..., 1] * (H - 4) + 2

        # Warmup CPU
        for _ in range(2):
            mast3r_slam_cpu_backends.iter_proj(
                rays_img.contiguous(),
                pts_3d.contiguous(),
                p_init.contiguous(),
                10,
                1.0,
                1e-4,
            )

        # Benchmark CPU
        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.iter_proj(
                rays_img.contiguous(),
                pts_3d.contiguous(),
                p_init.contiguous(),
                10,
                1.0,
                1e-4,
            )
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = sum(cpu_times) / len(cpu_times) * 1000

        # Warmup Metal
        for _ in range(2):
            mast3r_slam_metal_backends.iter_proj(
                rays_img.contiguous(),
                pts_3d.contiguous(),
                p_init.contiguous(),
                10,
                1.0,
                1e-4,
            )

        # Benchmark Metal
        metal_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_metal_backends.iter_proj(
                rays_img.contiguous(),
                pts_3d.contiguous(),
                p_init.contiguous(),
                10,
                1.0,
                1e-4,
            )
            metal_times.append(time.perf_counter() - start)
        metal_avg = sum(metal_times) / len(metal_times) * 1000

        speedup = cpu_avg / metal_avg
        winner = "Metal" if speedup > 1 else "CPU"

        print(
            f"{n_points:>10} {H}x{W:>7} {cpu_avg:>12.2f} {metal_avg:>12.2f} {speedup:>9.2f}x ({winner})"
        )


def benchmark_throughput():
    """Measure raw computational throughput."""
    print("\n" + "=" * 70)
    print("THROUGHPUT ANALYSIS")
    print("=" * 70)

    # Large scale for throughput measurement
    n_points = 100000
    H, W = 512, 512
    fdim = 24
    radius, dilation_max = 2, 2
    search_area = (2 * radius * dilation_max + 1) ** 2  # 81 positions

    D11 = torch.nn.functional.normalize(
        torch.randn(1, H, W, fdim, dtype=torch.float32), dim=-1
    )
    D21 = torch.nn.functional.normalize(
        torch.randn(1, n_points, fdim, dtype=torch.float32), dim=-1
    )
    p1 = torch.zeros(1, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (1, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (1, n_points))

    # Total operations: n_points * search_area * fdim * 2 (mul + add)
    total_flops = n_points * search_area * fdim * 2

    # Benchmark CPU
    for _ in range(3):
        mast3r_slam_cpu_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )

    n_runs = 20
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        mast3r_slam_cpu_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = sum(cpu_times) / len(cpu_times)

    # Benchmark Metal
    for _ in range(3):
        mast3r_slam_metal_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )

    metal_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        mast3r_slam_metal_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )
        metal_times.append(time.perf_counter() - start)
    metal_avg = sum(metal_times) / len(metal_times)

    cpu_gflops = total_flops / cpu_avg / 1e9
    metal_gflops = total_flops / metal_avg / 1e9

    print(
        f"\n  Configuration: {n_points} points, {H}x{W} image, {fdim} dim descriptors"
    )
    print(f"  Search area: {search_area} positions per point")
    print(f"  Total FLOPs: {total_flops / 1e9:.2f} GFLOP")
    print(f"\n  CPU:")
    print(f"    Time: {cpu_avg * 1000:.2f} ms")
    print(f"    Throughput: {cpu_gflops:.1f} GFLOPS")
    print(f"\n  Metal:")
    print(f"    Time: {metal_avg * 1000:.2f} ms")
    print(f"    Throughput: {metal_gflops:.1f} GFLOPS")
    print(f"\n  Note: Metal throughput includes CPU-GPU copy overhead.")
    print(f"        Pure GPU throughput would be significantly higher.")


def main():
    print("=" * 70)
    print("MASt3R-SLAM Backend Benchmarks")
    print("=" * 70)

    mast3r_slam_metal_backends.initialize()

    benchmark_refine_matches()
    benchmark_iter_proj()
    benchmark_throughput()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
For small workloads (<20K points), CPU is faster due to:
- Copy overhead CPU -> Metal -> CPU
- Metal kernel launch overhead

For large workloads (>50K points), Metal approaches CPU performance.
Pure GPU performance (without copies) would be 5-10x faster.

To fully leverage Metal:
1. Keep data on GPU between operations
2. Batch multiple operations together
3. Use MPS tensors directly (future optimization)
""")


if __name__ == "__main__":
    main()
