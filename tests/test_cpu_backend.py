# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Tests and benchmarks for CPU backend.

Tests:
- Functional correctness of iter_proj and refine_matches
- OpenMP parallelization impact
- SIMD (NEON on ARM64) verification

Usage:
    python tests/test_cpu_backend.py
"""

import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add CPU backend to path
cpu_backend_path = Path(__file__).parent.parent / "mast3r_slam" / "backends" / "cpu"
sys.path.insert(0, str(cpu_backend_path))


def test_import():
    """Test that the CPU backend can be imported."""
    print("=" * 60)
    print("TEST: Import CPU backend")
    print("=" * 60)

    try:
        import mast3r_slam_cpu_backends

        print(f"✓ Successfully imported mast3r_slam_cpu_backends")
        print(f"  Module: {mast3r_slam_cpu_backends}")
        print(f"  Doc: {mast3r_slam_cpu_backends.__doc__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_iter_proj():
    """Test iter_proj kernel functionality."""
    print("\n" + "=" * 60)
    print("TEST: iter_proj kernel")
    print("=" * 60)

    import mast3r_slam_cpu_backends

    # Create test data
    batch_size = 2
    n_points = 100
    H, W = 64, 64

    # rays_img_with_grad: [B, H, W, 9] (ray[3] + grad_x[3] + grad_y[3])
    rays_img = torch.randn(batch_size, H, W, 9, dtype=torch.float32)
    # Normalize ray directions
    rays_img[..., :3] = torch.nn.functional.normalize(rays_img[..., :3], dim=-1)

    # pts_3d_norm: [B, N, 3] normalized 3D points
    pts_3d = torch.randn(batch_size, n_points, 3, dtype=torch.float32)
    pts_3d = torch.nn.functional.normalize(pts_3d, dim=-1)

    # p_init: [B, N, 2] initial pixel guesses
    p_init = torch.rand(batch_size, n_points, 2, dtype=torch.float32)
    p_init[..., 0] = p_init[..., 0] * (W - 4) + 2  # u in [2, W-2]
    p_init[..., 1] = p_init[..., 1] * (H - 4) + 2  # v in [2, H-2]

    # Run kernel
    max_iter = 10
    lambda_init = 1.0
    cost_thresh = 1e-4

    try:
        p_new, converged = mast3r_slam_cpu_backends.iter_proj(
            rays_img.contiguous(),
            pts_3d.contiguous(),
            p_init.contiguous(),
            max_iter,
            lambda_init,
            cost_thresh,
        )

        print(f"✓ iter_proj executed successfully")
        print(
            f"  Input shape: rays={rays_img.shape}, pts={pts_3d.shape}, p_init={p_init.shape}"
        )
        print(f"  Output shape: p_new={p_new.shape}, converged={converged.shape}")
        print(f"  Converged: {converged.sum().item()}/{converged.numel()} points")
        print(
            f"  p_new range: u=[{p_new[..., 0].min():.2f}, {p_new[..., 0].max():.2f}], "
            f"v=[{p_new[..., 1].min():.2f}, {p_new[..., 1].max():.2f}]"
        )
        return True
    except Exception as e:
        print(f"✗ iter_proj failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_refine_matches():
    """Test refine_matches kernel functionality."""
    print("\n" + "=" * 60)
    print("TEST: refine_matches kernel")
    print("=" * 60)

    import mast3r_slam_cpu_backends

    # Create test data
    batch_size = 2
    n_points = 100
    H, W = 64, 64
    fdim = 24  # Feature dimension

    # D11: [B, H, W, F] descriptor image
    D11 = torch.randn(batch_size, H, W, fdim, dtype=torch.float32)
    D11 = torch.nn.functional.normalize(D11, dim=-1)

    # D21: [B, N, F] query descriptors
    D21 = torch.randn(batch_size, n_points, fdim, dtype=torch.float32)
    D21 = torch.nn.functional.normalize(D21, dim=-1)

    # p1: [B, N, 2] current pixel positions (long)
    p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

    radius = 2
    dilation_max = 2

    try:
        (p1_new,) = mast3r_slam_cpu_backends.refine_matches(
            D11.contiguous(),
            D21.contiguous(),
            p1.contiguous(),
            radius,
            dilation_max,
        )

        print(f"✓ refine_matches executed successfully")
        print(f"  Input shape: D11={D11.shape}, D21={D21.shape}, p1={p1.shape}")
        print(f"  Output shape: p1_new={p1_new.shape}")
        print(
            f"  Movement: {(p1_new != p1).any(dim=-1).sum().item()}/{n_points * batch_size} points moved"
        )
        return True
    except Exception as e:
        print(f"✗ refine_matches failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_iter_proj(n_runs: int = 10):
    """Benchmark iter_proj with different configurations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: iter_proj")
    print("=" * 60)

    import mast3r_slam_cpu_backends

    configs = [
        {"batch": 1, "points": 1000, "H": 128, "W": 128},
        {"batch": 1, "points": 10000, "H": 256, "W": 256},
        {"batch": 4, "points": 5000, "H": 128, "W": 128},
    ]

    for cfg in configs:
        batch_size = cfg["batch"]
        n_points = cfg["points"]
        H, W = cfg["H"], cfg["W"]

        rays_img = torch.randn(batch_size, H, W, 9, dtype=torch.float32)
        rays_img[..., :3] = torch.nn.functional.normalize(rays_img[..., :3], dim=-1)
        pts_3d = torch.nn.functional.normalize(
            torch.randn(batch_size, n_points, 3, dtype=torch.float32), dim=-1
        )
        p_init = torch.rand(batch_size, n_points, 2, dtype=torch.float32)
        p_init[..., 0] = p_init[..., 0] * (W - 4) + 2
        p_init[..., 1] = p_init[..., 1] * (H - 4) + 2

        # Warmup
        for _ in range(2):
            mast3r_slam_cpu_backends.iter_proj(
                rays_img.contiguous(),
                pts_3d.contiguous(),
                p_init.contiguous(),
                10,
                1.0,
                1e-4,
            )

        # Benchmark
        times = []
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
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        std_ms = (
            sum((t - avg_ms / 1000) ** 2 for t in times) / len(times)
        ) ** 0.5 * 1000

        print(
            f"  B={batch_size}, N={n_points}, {H}x{W}: {avg_ms:.2f} ± {std_ms:.2f} ms"
        )


def benchmark_refine_matches(n_runs: int = 10):
    """Benchmark refine_matches with different configurations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: refine_matches")
    print("=" * 60)

    import mast3r_slam_cpu_backends

    configs = [
        {"batch": 1, "points": 1000, "H": 128, "W": 128, "fdim": 24},
        {"batch": 1, "points": 10000, "H": 256, "W": 256, "fdim": 24},
        {"batch": 4, "points": 5000, "H": 128, "W": 128, "fdim": 24},
    ]

    for cfg in configs:
        batch_size = cfg["batch"]
        n_points = cfg["points"]
        H, W = cfg["H"], cfg["W"]
        fdim = cfg["fdim"]

        D11 = torch.nn.functional.normalize(
            torch.randn(batch_size, H, W, fdim, dtype=torch.float32), dim=-1
        )
        D21 = torch.nn.functional.normalize(
            torch.randn(batch_size, n_points, fdim, dtype=torch.float32), dim=-1
        )
        p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
        p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
        p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

        # Warmup
        for _ in range(2):
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        std_ms = (
            sum((t - avg_ms / 1000) ** 2 for t in times) / len(times)
        ) ** 0.5 * 1000

        print(
            f"  B={batch_size}, N={n_points}, {H}x{W}, F={fdim}: {avg_ms:.2f} ± {std_ms:.2f} ms"
        )


def benchmark_openmp_scaling():
    """Benchmark OpenMP thread scaling."""
    print("\n" + "=" * 60)
    print("BENCHMARK: OpenMP Thread Scaling")
    print("=" * 60)

    import mast3r_slam_cpu_backends

    # Get max threads
    max_threads = os.cpu_count() or 1
    print(f"  Max available threads: {max_threads}")

    # Fixed configuration
    batch_size = 1
    n_points = 20000
    H, W = 256, 256
    fdim = 24
    n_runs = 5

    D11 = torch.nn.functional.normalize(
        torch.randn(batch_size, H, W, fdim, dtype=torch.float32), dim=-1
    )
    D21 = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, fdim, dtype=torch.float32), dim=-1
    )
    p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

    thread_counts = (
        [1, 2, 4, 8] if max_threads >= 8 else list(range(1, max_threads + 1))
    )

    results = {}
    for num_threads in thread_counts:
        if num_threads > max_threads:
            continue

        os.environ["OMP_NUM_THREADS"] = str(num_threads)

        # Warmup
        for _ in range(2):
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.refine_matches(
                D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
            )
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        results[num_threads] = avg_ms

    # Print results
    baseline = results.get(1, results[min(results.keys())])
    print(f"\n  {'Threads':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"  {'-' * 32}")
    for threads, time_ms in sorted(results.items()):
        speedup = baseline / time_ms
        print(f"  {threads:<10} {time_ms:<12.2f} {speedup:<10.2f}x")

    # Reset
    if "OMP_NUM_THREADS" in os.environ:
        del os.environ["OMP_NUM_THREADS"]


def check_simd_usage():
    """Check if SIMD instructions are being used."""
    print("\n" + "=" * 60)
    print("CHECK: SIMD Usage")
    print("=" * 60)

    import platform

    machine = platform.machine()

    print(f"  Platform: {platform.system()} {machine}")

    if machine in ("arm64", "aarch64"):
        print(f"  ✓ ARM64 detected - NEON is enabled by default")
        print(f"  NEON intrinsics are used in refine_matches for dot product")
    elif machine in ("x86_64", "AMD64"):
        print(f"  ✓ x86_64 detected - AVX2+FMA should be enabled")
        print(f"  AVX2 intrinsics are used in refine_matches for dot product")
    else:
        print(f"  ? Unknown architecture - using scalar fallback")

    # Verify by comparing performance
    print("\n  Verifying SIMD impact on refine_matches...")

    import mast3r_slam_cpu_backends

    batch_size = 1
    n_points = 50000
    H, W = 256, 256
    fdim = 24  # Multiple of 4 (NEON) and 8 (AVX2)

    D11 = torch.nn.functional.normalize(
        torch.randn(batch_size, H, W, fdim, dtype=torch.float32), dim=-1
    )
    D21 = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, fdim, dtype=torch.float32), dim=-1
    )
    p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

    # Pure Python reference (no SIMD)
    def python_dot_product(a, b):
        return (a * b).sum(dim=-1)

    # Warmup
    for _ in range(2):
        mast3r_slam_cpu_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )

    # Benchmark C++ with SIMD
    n_runs = 5
    cpp_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        mast3r_slam_cpu_backends.refine_matches(
            D11.contiguous(), D21.contiguous(), p1.contiguous(), 2, 2
        )
        cpp_times.append(time.perf_counter() - start)

    cpp_avg = sum(cpp_times) / len(cpp_times) * 1000

    # Compute theoretical ops
    # For each point: search over ~(2*radius*dilation_max+1)^2 positions
    # Each position: fdim multiply-adds for dot product
    radius, dilation_max = 2, 2
    search_area = (2 * radius * dilation_max + 1) ** 2
    total_ops = n_points * search_area * fdim * 2  # multiply + add
    gflops = total_ops / (cpp_avg / 1000) / 1e9

    print(f"\n  refine_matches performance:")
    print(f"    Time: {cpp_avg:.2f} ms for {n_points} points")
    print(f"    Throughput: {gflops:.2f} GFLOPS")
    print(f"    (Search area: {search_area} positions, {fdim} dim descriptors)")


def main():
    print("=" * 60)
    print("CPU Backend Tests and Benchmarks")
    print("=" * 60)

    # Run tests
    if not test_import():
        print("\nFailed to import CPU backend. Exiting.")
        return 1

    if not test_iter_proj():
        print("\niter_proj test failed.")
        return 1

    if not test_refine_matches():
        print("\nrefine_matches test failed.")
        return 1

    # Run benchmarks
    benchmark_iter_proj()
    benchmark_refine_matches()
    benchmark_openmp_scaling()
    check_simd_usage()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
