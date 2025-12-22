# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
"""
Build script for Metal backend extension.

Usage:
    cd mast3r_slam/backends/metal
    python setup.py build_ext --inplace

Requirements:
    - macOS with Metal support
    - Xcode Command Line Tools
"""

import os
import platform
import sys

if platform.system() != "Darwin":
    print("Metal backend is only available on macOS")
    sys.exit(1)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# Metal/Obj-C++ specific flags
extra_compile_args = [
    "-O3",
    "-std=c++17",
    "-fobjc-arc",  # Automatic Reference Counting
]

extra_link_args = [
    "-framework",
    "Metal",
    "-framework",
    "Foundation",
]

setup(
    name="mast3r_slam_metal_backends",
    ext_modules=[
        CppExtension(
            "mast3r_slam_metal_backends",
            sources=[
                "metal_ops_optimized.mm",  # Use optimized version
                "bindings.cpp",
            ],
            include_dirs=[ROOT],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="objc++",
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
