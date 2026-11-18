#define _POSIX_C_SOURCE 200112L

#include "simd_matmul.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#endif

typedef void (*kernel_fn)(const float *, const float *, float *, size_t);

typedef struct {
    const char *name;
    double seconds;
    double gflops;
    double checksum;
    float max_error;
} result_t;

static double now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
#endif
}

static void *aligned_alloc_32(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 32);
#else
    void *ptr = NULL;
    return posix_memalign(&ptr, 32, bytes) == 0 ? ptr : NULL;
#endif
}

static void aligned_free_32(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static uint32_t random_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_matrix(float *matrix, size_t count, uint32_t *seed) {
    size_t i;
    for (i = 0; i < count; ++i) {
        matrix[i] = ((float)(random_u32(seed) & 0xffffu) / 32768.0f) - 1.0f;
    }
}

static double checksum(const float *matrix, size_t count) {
    double sum = 0.0;
    size_t i;
    for (i = 0; i < count; ++i) {
        sum += matrix[i];
    }
    return sum;
}

static float max_abs_diff(const float *actual, const float *expected, size_t count) {
    float result = 0.0f;
    size_t i;
    for (i = 0; i < count; ++i) {
        const float error = fabsf(actual[i] - expected[i]);
        if (error > result) {
            result = error;
        }
    }
    return result;
}

static result_t run_kernel(const char *name, kernel_fn kernel,
                           const float *a, const float *b, float *c,
                           const float *reference, size_t n, size_t iterations) {
    result_t result;
    size_t iteration;
    const size_t count = n * n;

    result.name = name;
    result.seconds = DBL_MAX;
    kernel(a, b, c, n); /* Warm code and data before recording the best time. */

    for (iteration = 0; iteration < iterations; ++iteration) {
        const double start = now_seconds();
        kernel(a, b, c, n);
        {
            const double elapsed = now_seconds() - start;
            if (elapsed < result.seconds) {
                result.seconds = elapsed;
            }
        }
    }

    result.gflops = (2.0 * (double)n * (double)n * (double)n) /
                    (result.seconds * 1.0e9);
    result.checksum = checksum(c, count);
    result.max_error = reference == NULL ? 0.0f : max_abs_diff(c, reference, count);
    return result;
}

static int parse_size(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > 8192) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static void usage(const char *program) {
    fprintf(stderr, "Usage: %s [--size N] [--iterations N] [--csv FILE]\n", program);
}

int main(int argc, char **argv) {
    size_t n = 512;
    size_t iterations = 3;
    const char *csv_path = NULL;
    size_t count;
    size_t bytes;
    float *a;
    float *b;
    float *c;
    float *reference;
    uint32_t seed = 0x12345678u;
    result_t results[4];
    size_t result_count = 0;
    size_t i;
    FILE *csv = NULL;

    for (i = 1; i < (size_t)argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < (size_t)argc) {
            if (!parse_size(argv[++i], &n)) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < (size_t)argc) {
            if (!parse_size(argv[++i], &iterations)) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < (size_t)argc) {
            csv_path = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Matrix size overflows address space.\n");
        return 2;
    }
    count = n * n;
    bytes = count * sizeof(float);
    a = (float *)aligned_alloc_32(bytes);
    b = (float *)aligned_alloc_32(bytes);
    c = (float *)aligned_alloc_32(bytes);
    reference = (float *)aligned_alloc_32(bytes);
    if (a == NULL || b == NULL || c == NULL || reference == NULL) {
        fprintf(stderr, "Could not allocate four %zu-byte matrices.\n", bytes);
        aligned_free_32(a);
        aligned_free_32(b);
        aligned_free_32(c);
        aligned_free_32(reference);
        return 1;
    }

    fill_matrix(a, count, &seed);
    fill_matrix(b, count, &seed);

    results[result_count] = run_kernel("naive-ijk", matmul_naive, a, b, c,
                                       NULL, n, iterations);
    memcpy(reference, c, bytes);
    ++result_count;
    results[result_count++] = run_kernel("scalar-ikj", matmul_scalar, a, b, c,
                                         reference, n, iterations);
    if (simd_avx2_fma_supported()) {
        results[result_count++] = run_kernel("avx2-fma-ikj", matmul_avx2, a, b, c,
                                             reference, n, iterations);
        results[result_count++] = run_kernel("avx2-blocked", matmul_blocked_avx2,
                                             a, b, c, reference, n, iterations);
    }

    printf("SGEMM: C = A x B, N=%zu, best of %zu timed runs\n", n, iterations);
    printf("%-15s %10s %10s %12s %12s\n",
           "kernel", "time-ms", "GFLOP/s", "vs-naive", "max-error");
    for (i = 0; i < result_count; ++i) {
        printf("%-15s %10.3f %10.2f %11.2fx %12.6g\n",
               results[i].name,
               results[i].seconds * 1.0e3,
               results[i].gflops,
               results[0].seconds / results[i].seconds,
               (double)results[i].max_error);
    }
    if (result_count < 4) {
        printf("AVX2/FMA kernels skipped: CPU or OS support was not detected.\n");
    } else {
        printf("SIMD-only speedup (same ikj access pattern): %.2fx\n",
               results[1].seconds / results[2].seconds);
        printf("Blocking gain over unblocked AVX2: %.2fx\n",
               results[2].seconds / results[3].seconds);
    }
    printf("checksum: %.9g\n", results[result_count - 1].checksum);

    if (csv_path != NULL) {
        csv = fopen(csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "Could not open CSV output: %s\n", csv_path);
            aligned_free_32(a);
            aligned_free_32(b);
            aligned_free_32(c);
            aligned_free_32(reference);
            return 1;
        }
        fprintf(csv, "size,kernel,seconds,gflops,speedup_vs_naive,max_error,checksum\n");
        for (i = 0; i < result_count; ++i) {
            fprintf(csv, "%zu,%s,%.9f,%.6f,%.6f,%.9g,%.12g\n",
                    n, results[i].name, results[i].seconds, results[i].gflops,
                    results[0].seconds / results[i].seconds,
                    (double)results[i].max_error, results[i].checksum);
        }
        fclose(csv);
        printf("Wrote %s\n", csv_path);
    }

    aligned_free_32(a);
    aligned_free_32(b);
    aligned_free_32(c);
    aligned_free_32(reference);
    return 0;
}
