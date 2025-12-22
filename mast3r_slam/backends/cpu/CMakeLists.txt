# Copyright Delanoe Pirard / Aedelon. Apache 2.0 License.
# CMake build for CPU backend with OpenMP + SIMD

cmake_minimum_required(VERSION 3.18)
project(mast3r_slam_cpu_backends LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Detect architecture for SIMD
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # x86_64: Try AVX2
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(SIMD_FLAGS "-mavx2 -mfma")
        message(STATUS "Enabling AVX2 + FMA SIMD optimizations")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    # ARM64: NEON is always available
    set(SIMD_FLAGS "")
    message(STATUS "ARM64 detected, NEON enabled by default")
endif()

# Create the Python extension
pybind11_add_module(mast3r_slam_cpu_backends
    src/cpu_kernels.cpp
)

# Link libraries
target_link_libraries(mast3r_slam_cpu_backends PRIVATE
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Compiler flags
target_compile_options(mast3r_slam_cpu_backends PRIVATE
    -O3
    -ffast-math
    ${SIMD_FLAGS}
)

# Include directories
target_include_directories(mast3r_slam_cpu_backends PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

# Set output directory
set_target_properties(mast3r_slam_cpu_backends PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
