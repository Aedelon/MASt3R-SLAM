# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Build script for CPU backend extension.

Usage:
    cd mast3r_slam/backends/cpu
    python setup.py build_ext --inplace
"""

import os
import platform
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Detect platform and set compiler flags
extra_compile_args = ["-O3", "-ffast-math"]
extra_link_args = []

system = platform.system()
machine = platform.machine()

# OpenMP flags
if system == "Darwin":
    # macOS: Use libomp from Homebrew
    omp_include = "/opt/homebrew/opt/libomp/include"
    omp_lib = "/opt/homebrew/opt/libomp/lib"
    if os.path.exists(omp_include):
        extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"]
        extra_link_args += [f"-L{omp_lib}", "-lomp"]
    else:
        # Try Intel path
        omp_include = "/usr/local/opt/libomp/include"
        omp_lib = "/usr/local/opt/libomp/lib"
        if os.path.exists(omp_include):
            extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"]
            extra_link_args += [f"-L{omp_lib}", "-lomp"]
        else:
            print("Warning: OpenMP not found. Building without parallelization.")
else:
    # Linux/Windows: Standard OpenMP
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

# SIMD flags
if machine in ("x86_64", "AMD64"):
    extra_compile_args += ["-mavx2", "-mfma"]
    print("Enabling AVX2 + FMA SIMD optimizations")
elif machine in ("arm64", "aarch64"):
    # NEON is enabled by default on ARM64
    print("ARM64 detected, NEON enabled by default")

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="mast3r_slam_cpu_backends",
    ext_modules=[
        CppExtension(
            "mast3r_slam_cpu_backends",
            sources=[
                "src/matching.cpp",
                "src/gn.cpp",
                "src/bindings.cpp",
            ],
            include_dirs=[os.path.join(ROOT, "include")],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
