# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Tests and benchmarks for Metal backend.

Usage:
    python tests/test_metal_backend.py
"""

import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add Metal backend to path
metal_backend_path = Path(__file__).parent.parent / "mast3r_slam" / "backends" / "metal"
sys.path.insert(0, str(metal_backend_path))


def test_import():
    """Test that the Metal backend can be imported."""
    print("=" * 60)
    print("TEST: Import Metal backend")
    print("=" * 60)

    try:
        import mast3r_slam_metal_backends

        print(f"✓ Successfully imported mast3r_slam_metal_backends")
        print(f"  Module: {mast3r_slam_metal_backends}")
        print(f"  Doc: {mast3r_slam_metal_backends.__doc__}")

        if mast3r_slam_metal_backends.is_available():
            print(f"  ✓ Metal is available")
        else:
            print(f"  ✗ Metal is NOT available")
            return False

        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_initialization():
    """Test Metal device initialization."""
    print("\n" + "=" * 60)
    print("TEST: Metal initialization")
    print("=" * 60)

    import mast3r_slam_metal_backends

    try:
        result = mast3r_slam_metal_backends.initialize()
        if result:
            print(f"✓ Metal initialized successfully")
            return True
        else:
            print(f"✗ Metal initialization failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_iter_proj():
    """Test iter_proj kernel functionality."""
    print("\n" + "=" * 60)
    print("TEST: iter_proj kernel")
    print("=" * 60)

    import mast3r_slam_metal_backends

    # Create test data
    batch_size = 2
    n_points = 100
    H, W = 64, 64

    # rays_img_with_grad: [B, H, W, 9]
    rays_img = torch.randn(batch_size, H, W, 9, dtype=torch.float32)
    rays_img[..., :3] = torch.nn.functional.normalize(rays_img[..., :3], dim=-1)

    # pts_3d_norm: [B, N, 3]
    pts_3d = torch.randn(batch_size, n_points, 3, dtype=torch.float32)
    pts_3d = torch.nn.functional.normalize(pts_3d, dim=-1)

    # p_init: [B, N, 2]
    p_init = torch.rand(batch_size, n_points, 2, dtype=torch.float32)
    p_init[..., 0] = p_init[..., 0] * (W - 4) + 2
    p_init[..., 1] = p_init[..., 1] * (H - 4) + 2

    try:
        result = mast3r_slam_metal_backends.iter_proj(
            rays_img.contiguous(),
            pts_3d.contiguous(),
            p_init.contiguous(),
            10,  # max_iter
            1.0,  # lambda_init
            1e-4,  # cost_thresh
        )

        if len(result) == 2:
            p_new, converged = result
            print(f"✓ iter_proj executed successfully")
            print(
                f"  Input: rays={rays_img.shape}, pts={pts_3d.shape}, p_init={p_init.shape}"
            )
            print(f"  Output: p_new={p_new.shape}, converged={converged.shape}")
            print(f"  Converged: {converged.sum().item()}/{converged.numel()} points")
            return True
        else:
            print(f"✗ Unexpected result length: {len(result)}")
            return False

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

    import mast3r_slam_metal_backends

    batch_size = 2
    n_points = 100
    H, W = 64, 64
    fdim = 24

    D11 = torch.randn(batch_size, H, W, fdim, dtype=torch.float32)
    D11 = torch.nn.functional.normalize(D11, dim=-1)

    D21 = torch.randn(batch_size, n_points, fdim, dtype=torch.float32)
    D21 = torch.nn.functional.normalize(D21, dim=-1)

    p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

    try:
        result = mast3r_slam_metal_backends.refine_matches(
            D11.contiguous(),
            D21.contiguous(),
            p1.contiguous(),
            2,  # radius
            2,  # dilation_max
        )

        if len(result) == 1:
            (p1_new,) = result
            print(f"✓ refine_matches executed successfully")
            print(f"  Input: D11={D11.shape}, D21={D21.shape}, p1={p1.shape}")
            print(f"  Output: p1_new={p1_new.shape}")
            moved = (p1_new != p1).any(dim=-1).sum().item()
            print(f"  Movement: {moved}/{n_points * batch_size} points moved")
            return True
        else:
            print(f"✗ Unexpected result length: {len(result)}")
            return False

    except Exception as e:
        print(f"✗ refine_matches failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_metal_vs_cpu():
    """Compare Metal vs CPU performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Metal vs CPU")
    print("=" * 60)

    import mast3r_slam_metal_backends

    # Check if CPU backend is also available
    try:
        sys.path.insert(
            0,
            str(Path(__file__).parent.parent / "mast3r_slam" / "backends" / "cpu"),
        )
        import mast3r_slam_cpu_backends

        has_cpu = True
    except ImportError:
        has_cpu = False
        print("  CPU backend not available for comparison")

    # Test configuration
    batch_size = 1
    n_points = 10000
    H, W = 256, 256
    fdim = 24
    n_runs = 10

    D11 = torch.nn.functional.normalize(
        torch.randn(batch_size, H, W, fdim, dtype=torch.float32), dim=-1
    )
    D21 = torch.nn.functional.normalize(
        torch.randn(batch_size, n_points, fdim, dtype=torch.float32), dim=-1
    )
    p1 = torch.zeros(batch_size, n_points, 2, dtype=torch.int64)
    p1[..., 0] = torch.randint(5, W - 5, (batch_size, n_points))
    p1[..., 1] = torch.randint(5, H - 5, (batch_size, n_points))

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

    print(f"\n  refine_matches ({n_points} points, {H}x{W}):")
    print(f"    Metal: {metal_avg:.2f} ms")

    if has_cpu:
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
        speedup = cpu_avg / metal_avg

        print(f"    CPU:   {cpu_avg:.2f} ms")
        print(f"    Speedup: {speedup:.2f}x")


def main():
    print("=" * 60)
    print("Metal Backend Tests and Benchmarks")
    print("=" * 60)

    if not test_import():
        print("\nFailed to import Metal backend. Exiting.")
        return 1

    if not test_initialization():
        print("\nMetal initialization failed. Exiting.")
        return 1

    if not test_iter_proj():
        print("\niter_proj test failed.")
        return 1

    if not test_refine_matches():
        print("\nrefine_matches test failed.")
        return 1

    benchmark_metal_vs_cpu()

    print("\n" + "=" * 60)
    print("All Metal tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
