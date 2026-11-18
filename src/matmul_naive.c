#include "simd_matmul.h"

void matmul_naive(const float *a, const float *b, float *c, size_t n) {
    size_t i;
    size_t j;
    size_t k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
