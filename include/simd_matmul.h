#ifndef SIMD_MATMUL_H
#define SIMD_MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matrices are row-major, square, and may alias neither inputs nor outputs. */
void matmul_naive(const float *a, const float *b, float *c, size_t n);
void matmul_scalar(const float *a, const float *b, float *c, size_t n);
void matmul_avx2(const float *a, const float *b, float *c, size_t n);
void matmul_blocked_avx2(const float *a, const float *b, float *c, size_t n);

/* The AVX2 kernel also uses FMA and requires OS support for AVX state. */
int simd_avx2_fma_supported(void);

#ifdef __cplusplus
}
#endif

#endif
