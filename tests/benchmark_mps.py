# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Benchmark comparing CPU tensors vs MPS tensors (zero-copy) with Metal backend.
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


def benchmark_refine_matches_mps(n_runs: int = 10):
    """Benchmark refine_matches: CPU vs Metal(CPU) vs Metal(MPS)."""
    print("=" * 70)
    print("BENCHMARK: refine_matches - CPU vs Metal(CPU) vs Metal(MPS)")
    print("=" * 70)

    if not torch.backends.mps.is_available():
        print("MPS not available, skipping MPS benchmarks")
        return

    configs = [
        {"n_points": 10000, "H": 256, "W": 256, "fdim": 24},
        {"n_points": 50000, "H": 512, "W": 512, "fdim": 24},
        {"n_points": 100000, "H": 512, "W": 512, "fdim": 24},
    ]

    print(
        f"\n{'Points':>10} {'CPU (ms)':>12} {'Metal/CPU':>12} {'Metal/MPS':>12} {'Speedup':>10}"
    )
    print("-" * 60)

    for cfg in configs:
        n_points = cfg["n_points"]
        H, W = cfg["H"], cfg["W"]
        fdim = cfg["fdim"]

        # Create test data on CPU
        D11_cpu = torch.nn.functional.normalize(
            torch.randn(1, H, W, fdim, dtype=torch.float32), dim=-1
        )
        D21_cpu = torch.nn.functional.normalize(
            torch.randn(1, n_points, fdim, dtype=torch.float32), dim=-1
        )
        p1_cpu = torch.zeros(1, n_points, 2, dtype=torch.int64)
        p1_cpu[..., 0] = torch.randint(5, W - 5, (1, n_points))
        p1_cpu[..., 1] = torch.randint(5, H - 5, (1, n_points))

        # Create MPS tensors
        D11_mps = D11_cpu.to("mps")
        D21_mps = D21_cpu.to("mps")
        p1_mps = p1_cpu.to("mps")
        torch.mps.synchronize()

        # Warmup CPU backend
        for _ in range(2):
            mast3r_slam_cpu_backends.refine_matches(
                D11_cpu.contiguous(), D21_cpu.contiguous(), p1_cpu.contiguous(), 2, 2
            )

        # Benchmark CPU backend
        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.refine_matches(
                D11_cpu.contiguous(), D21_cpu.contiguous(), p1_cpu.contiguous(), 2, 2
            )
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = sum(cpu_times) / len(cpu_times) * 1000

        # Warmup Metal with CPU tensors
        for _ in range(2):
            mast3r_slam_metal_backends.refine_matches(
                D11_cpu.contiguous(), D21_cpu.contiguous(), p1_cpu.contiguous(), 2, 2
            )

        # Benchmark Metal with CPU tensors
        metal_cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_metal_backends.refine_matches(
                D11_cpu.contiguous(), D21_cpu.contiguous(), p1_cpu.contiguous(), 2, 2
            )
            metal_cpu_times.append(time.perf_counter() - start)
        metal_cpu_avg = sum(metal_cpu_times) / len(metal_cpu_times) * 1000

        # Warmup Metal with MPS tensors
        for _ in range(2):
            torch.mps.synchronize()
            mast3r_slam_metal_backends.refine_matches(
                D11_mps.contiguous(), D21_mps.contiguous(), p1_mps.contiguous(), 2, 2
            )
            torch.mps.synchronize()

        # Benchmark Metal with MPS tensors (zero-copy)
        metal_mps_times = []
        for _ in range(n_runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            mast3r_slam_metal_backends.refine_matches(
                D11_mps.contiguous(), D21_mps.contiguous(), p1_mps.contiguous(), 2, 2
            )
            torch.mps.synchronize()
            metal_mps_times.append(time.perf_counter() - start)
        metal_mps_avg = sum(metal_mps_times) / len(metal_mps_times) * 1000

        speedup_vs_cpu = cpu_avg / metal_mps_avg
        speedup_vs_metal_cpu = metal_cpu_avg / metal_mps_avg

        print(
            f"{n_points:>10} {cpu_avg:>12.2f} {metal_cpu_avg:>12.2f} {metal_mps_avg:>12.2f} "
            f"{speedup_vs_cpu:>9.2f}x"
        )


def benchmark_iter_proj_mps(n_runs: int = 10):
    """Benchmark iter_proj: CPU vs Metal(CPU) vs Metal(MPS)."""
    print("\n" + "=" * 70)
    print("BENCHMARK: iter_proj - CPU vs Metal(CPU) vs Metal(MPS)")
    print("=" * 70)

    if not torch.backends.mps.is_available():
        print("MPS not available, skipping MPS benchmarks")
        return

    configs = [
        {"n_points": 10000, "H": 256, "W": 256},
        {"n_points": 50000, "H": 512, "W": 512},
    ]

    print(
        f"\n{'Points':>10} {'CPU (ms)':>12} {'Metal/CPU':>12} {'Metal/MPS':>12} {'Speedup':>10}"
    )
    print("-" * 60)

    for cfg in configs:
        n_points = cfg["n_points"]
        H, W = cfg["H"], cfg["W"]

        # Create CPU tensors
        rays_img_cpu = torch.randn(1, H, W, 9, dtype=torch.float32)
        rays_img_cpu[..., :3] = torch.nn.functional.normalize(
            rays_img_cpu[..., :3], dim=-1
        )
        pts_3d_cpu = torch.nn.functional.normalize(
            torch.randn(1, n_points, 3, dtype=torch.float32), dim=-1
        )
        p_init_cpu = torch.rand(1, n_points, 2, dtype=torch.float32)
        p_init_cpu[..., 0] = p_init_cpu[..., 0] * (W - 4) + 2
        p_init_cpu[..., 1] = p_init_cpu[..., 1] * (H - 4) + 2

        # Create MPS tensors
        rays_img_mps = rays_img_cpu.to("mps")
        pts_3d_mps = pts_3d_cpu.to("mps")
        p_init_mps = p_init_cpu.to("mps")
        torch.mps.synchronize()

        # Benchmark CPU backend
        for _ in range(2):
            mast3r_slam_cpu_backends.iter_proj(
                rays_img_cpu.contiguous(),
                pts_3d_cpu.contiguous(),
                p_init_cpu.contiguous(),
                10,
                1.0,
                1e-4,
            )

        cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_cpu_backends.iter_proj(
                rays_img_cpu.contiguous(),
                pts_3d_cpu.contiguous(),
                p_init_cpu.contiguous(),
                10,
                1.0,
                1e-4,
            )
            cpu_times.append(time.perf_counter() - start)
        cpu_avg = sum(cpu_times) / len(cpu_times) * 1000

        # Benchmark Metal with CPU tensors
        for _ in range(2):
            mast3r_slam_metal_backends.iter_proj(
                rays_img_cpu.contiguous(),
                pts_3d_cpu.contiguous(),
                p_init_cpu.contiguous(),
                10,
                1.0,
                1e-4,
            )

        metal_cpu_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            mast3r_slam_metal_backends.iter_proj(
                rays_img_cpu.contiguous(),
                pts_3d_cpu.contiguous(),
                p_init_cpu.contiguous(),
                10,
                1.0,
                1e-4,
            )
            metal_cpu_times.append(time.perf_counter() - start)
        metal_cpu_avg = sum(metal_cpu_times) / len(metal_cpu_times) * 1000

        # Benchmark Metal with MPS tensors
        for _ in range(2):
            torch.mps.synchronize()
            mast3r_slam_metal_backends.iter_proj(
                rays_img_mps.contiguous(),
                pts_3d_mps.contiguous(),
                p_init_mps.contiguous(),
                10,
                1.0,
                1e-4,
            )
            torch.mps.synchronize()

        metal_mps_times = []
        for _ in range(n_runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            mast3r_slam_metal_backends.iter_proj(
                rays_img_mps.contiguous(),
                pts_3d_mps.contiguous(),
                p_init_mps.contiguous(),
                10,
                1.0,
                1e-4,
            )
            torch.mps.synchronize()
            metal_mps_times.append(time.perf_counter() - start)
        metal_mps_avg = sum(metal_mps_times) / len(metal_mps_times) * 1000

        speedup = cpu_avg / metal_mps_avg

        print(
            f"{n_points:>10} {cpu_avg:>12.2f} {metal_cpu_avg:>12.2f} {metal_mps_avg:>12.2f} "
            f"{speedup:>9.2f}x"
        )


def benchmark_batched_operations(n_runs: int = 10):
    """Benchmark batched operations: multiple pairs in single command buffer."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Batched Operations (Command Buffer Batching)")
    print("=" * 70)

    if not torch.backends.mps.is_available():
        print("MPS not available, skipping batched benchmarks")
        return

    # Test multiple refine_matches calls batched together
    n_pairs_list = [2, 4, 8]
    n_points = 10000
    H, W, fdim = 256, 256, 24

    print(
        f"\n{'Pairs':>6} {'Individual':>14} {'Batched':>14} {'Speedup':>10} {'Saving':>10}"
    )
    print("-" * 60)

    for n_pairs in n_pairs_list:
        # Create test data for each pair
        D11_list = []
        D21_list = []
        p1_list = []

        for _ in range(n_pairs):
            D11 = torch.nn.functional.normalize(
                torch.randn(1, H, W, fdim, dtype=torch.float32, device="mps"), dim=-1
            )
            D21 = torch.nn.functional.normalize(
                torch.randn(1, n_points, fdim, dtype=torch.float32, device="mps"), dim=-1
            )
            p1 = torch.zeros(1, n_points, 2, dtype=torch.int64, device="mps")
            p1[..., 0] = torch.randint(5, W - 5, (1, n_points), device="mps")
            p1[..., 1] = torch.randint(5, H - 5, (1, n_points), device="mps")
            D11_list.append(D11)
            D21_list.append(D21)
            p1_list.append(p1)

        torch.mps.synchronize()

        # Warmup individual calls
        for i in range(n_pairs):
            mast3r_slam_metal_backends.refine_matches(
                D11_list[i], D21_list[i], p1_list[i], 2, 2
            )
        torch.mps.synchronize()

        # Benchmark individual calls
        individual_times = []
        for _ in range(n_runs):
            torch.mps.synchronize()
            start = time.perf_counter()
            for i in range(n_pairs):
                mast3r_slam_metal_backends.refine_matches(
                    D11_list[i], D21_list[i], p1_list[i], 2, 2
                )
            torch.mps.synchronize()
            individual_times.append(time.perf_counter() - start)
        individual_avg = sum(individual_times) / len(individual_times) * 1000

        # Check if batched function exists
        if hasattr(mast3r_slam_metal_backends, "refine_matches_batched"):
            # Warmup batched
            mast3r_slam_metal_backends.refine_matches_batched(
                D11_list, D21_list, p1_list, 2, 2
            )
            torch.mps.synchronize()

            # Benchmark batched
            batched_times = []
            for _ in range(n_runs):
                torch.mps.synchronize()
                start = time.perf_counter()
                mast3r_slam_metal_backends.refine_matches_batched(
                    D11_list, D21_list, p1_list, 2, 2
                )
                torch.mps.synchronize()
                batched_times.append(time.perf_counter() - start)
            batched_avg = sum(batched_times) / len(batched_times) * 1000

            speedup = individual_avg / batched_avg
            saving = (individual_avg - batched_avg) / individual_avg * 100

            print(
                f"{n_pairs:>6} {individual_avg:>14.2f} {batched_avg:>14.2f} "
                f"{speedup:>9.2f}x {saving:>9.1f}%"
            )
        else:
            print(f"{n_pairs:>6} {individual_avg:>14.2f} {'N/A':>14} {'N/A':>10} {'N/A':>10}")
            print("  (refine_matches_batched not exported)")


def main():
    print("=" * 70)
    print("MPS Zero-Copy Benchmark")
    print("=" * 70)
    print(f"MPS available: {torch.backends.mps.is_available()}")

    mast3r_slam_metal_backends.initialize()

    benchmark_refine_matches_mps()
    benchmark_iter_proj_mps()
    benchmark_batched_operations()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Metal/MPS (zero-copy) eliminates CPU<->GPU data transfers.
This should show significant speedups vs Metal/CPU for all workloads.

The speedup column shows improvement vs CPU backend.
Compare Metal/CPU vs Metal/MPS to see zero-copy benefit.

Batched operations reduce synchronization overhead by combining
multiple kernel dispatches into a single command buffer.
""")


if __name__ == "__main__":
    main()
