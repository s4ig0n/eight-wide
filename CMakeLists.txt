cmake_minimum_required(VERSION 3.20)
project(simd_weekend VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(SIMD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(simd_kernels
    src/matmul_naive.c
    src/matmul_scalar.c
    src/matmul_avx2.c
    src/matmul_blocked_avx2.c
    src/cpu_features.c
)
target_include_directories(simd_kernels PUBLIC include)

if(MSVC)
    target_compile_options(simd_kernels PRIVATE /W4)
    set_source_files_properties(src/matmul_scalar.c PROPERTIES COMPILE_OPTIONS "/Qvec-")
    set_source_files_properties(
        src/matmul_avx2.c
        src/matmul_blocked_avx2.c
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
    if(SIMD_WARNINGS_AS_ERRORS)
        target_compile_options(simd_kernels PRIVATE /WX)
    endif()
else()
    target_compile_options(simd_kernels PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(src/matmul_scalar.c PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
    set_source_files_properties(
        src/matmul_avx2.c
        src/matmul_blocked_avx2.c
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
    if(SIMD_WARNINGS_AS_ERRORS)
        target_compile_options(simd_kernels PRIVATE -Werror)
    endif()
endif()

add_executable(simd_bench src/benchmark.c)
target_link_libraries(simd_bench PRIVATE simd_kernels)

if(NOT MSVC)
    target_link_libraries(simd_bench PRIVATE m)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(kernel_tests tests/test_kernels.c)
    target_link_libraries(kernel_tests PRIVATE simd_kernels)
    if(NOT MSVC)
        target_link_libraries(kernel_tests PRIVATE m)
    endif()
    add_test(NAME kernel_correctness COMMAND kernel_tests)
endif()
