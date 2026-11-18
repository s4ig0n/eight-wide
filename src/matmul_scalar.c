#include "simd_matmul.h"

#include <string.h>

/*
 * Same arithmetic as the AVX2 kernel, but scalar by construction. CMake turns
 * off the compiler's loop vectorizer for this translation unit so the
 * scalar-to-AVX2 comparison measures explicit SIMD rather than two versions
 * of compiler-generated SIMD.
 */
void matmul_scalar(const float *a, const float *b, float *c, size_t n) {
    size_t i;
    size_t j;
    size_t k;

    memset(c, 0, n * n * sizeof(*c));
    for (i = 0; i < n; ++i) {
        float *c_row = c + i * n;
        for (k = 0; k < n; ++k) {
            const float a_ik = a[i * n + k];
            const float *b_row = b + k * n;
            for (j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
