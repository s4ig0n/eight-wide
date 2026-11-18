#include "simd_matmul.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_size(size_t n) {
    const size_t count = n * n;
    float *a = (float *)malloc(count * sizeof(*a));
    float *b = (float *)malloc(count * sizeof(*b));
    float *expected = (float *)malloc(count * sizeof(*expected));
    float *actual = (float *)malloc(count * sizeof(*actual));
    size_t i;
    float max_error = 0.0f;

    if (a == NULL || b == NULL || expected == NULL || actual == NULL) {
        free(a);
        free(b);
        free(expected);
        free(actual);
        return 0;
    }
    for (i = 0; i < count; ++i) {
        a[i] = (float)((int)(i % 17) - 8) / 8.0f;
        b[i] = (float)((int)(i % 11) - 5) / 5.0f;
    }

    matmul_naive(a, b, expected, n);
    matmul_scalar(a, b, actual, n);
    for (i = 0; i < count; ++i) {
        const float error = fabsf(actual[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    if (max_error > 1.0e-4f * (float)n) {
        fprintf(stderr, "scalar failed for n=%zu: max error %.9g\n", n, (double)max_error);
        free(a); free(b); free(expected); free(actual);
        return 0;
    }

    if (simd_avx2_fma_supported()) {
        max_error = 0.0f;
        matmul_avx2(a, b, actual, n);
        for (i = 0; i < count; ++i) {
            const float error = fabsf(actual[i] - expected[i]);
            if (error > max_error) {
                max_error = error;
            }
        }
        if (max_error > 1.0e-4f * (float)n) {
            fprintf(stderr, "AVX2 failed for n=%zu: max error %.9g\n", n, (double)max_error);
            free(a); free(b); free(expected); free(actual);
            return 0;
        }
    }

    free(a);
    free(b);
    free(expected);
    free(actual);
    return 1;
}

int main(void) {
    static const size_t sizes[] = {1, 3, 8, 9, 16, 31};
    size_t i;
    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        if (!check_size(sizes[i])) {
            return 1;
        }
    }
    printf("kernel correctness tests passed\n");
    return 0;
}
