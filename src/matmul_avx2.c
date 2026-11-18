#include "simd_matmul.h"

#include <immintrin.h>
#include <string.h>

void matmul_avx2(const float *a, const float *b, float *c, size_t n) {
    size_t i;
    size_t j;
    size_t k;

    memset(c, 0, n * n * sizeof(*c));
    for (i = 0; i < n; ++i) {
        float *c_row = c + i * n;
        for (k = 0; k < n; ++k) {
            const __m256 a_vec = _mm256_set1_ps(a[i * n + k]);
            const float *b_row = b + k * n;

            for (j = 0; j + 8 <= n; j += 8) {
                const __m256 b_vec = _mm256_loadu_ps(b_row + j);
                __m256 c_vec = _mm256_loadu_ps(c_row + j);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(c_row + j, c_vec);
            }
            for (; j < n; ++j) {
                c_row[j] += a[i * n + k] * b_row[j];
            }
        }
    }
}
