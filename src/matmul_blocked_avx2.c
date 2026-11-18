#include "simd_matmul.h"

#include <immintrin.h>
#include <string.h>

#define BLOCK_SIZE 128

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

void matmul_blocked_avx2(const float *a, const float *b, float *c, size_t n) {
    size_t ii;
    size_t jj;
    size_t kk;

    memset(c, 0, n * n * sizeof(*c));
    for (ii = 0; ii < n; ii += BLOCK_SIZE) {
        const size_t i_end = min_size(ii + BLOCK_SIZE, n);
        for (kk = 0; kk < n; kk += BLOCK_SIZE) {
            const size_t k_end = min_size(kk + BLOCK_SIZE, n);
            for (jj = 0; jj < n; jj += BLOCK_SIZE) {
                const size_t j_end = min_size(jj + BLOCK_SIZE, n);
                size_t i;

                for (i = ii; i < i_end; ++i) {
                    float *c_row = c + i * n;
                    size_t k;

                    for (k = kk; k < k_end; ++k) {
                        const __m256 a_vec = _mm256_set1_ps(a[i * n + k]);
                        const float *b_row = b + k * n;
                        size_t j = jj;

                        for (; j + 8 <= j_end; j += 8) {
                            const __m256 b_vec = _mm256_loadu_ps(b_row + j);
                            __m256 c_vec = _mm256_loadu_ps(c_row + j);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(c_row + j, c_vec);
                        }
                        for (; j < j_end; ++j) {
                            c_row[j] += a[i * n + k] * b_row[j];
                        }
                    }
                }
            }
        }
    }
}
