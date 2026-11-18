#include "simd_matmul.h"

#if defined(_MSC_VER)
#include <intrin.h>

int simd_avx2_fma_supported(void) {
    int regs[4];
    unsigned __int64 xcr0;

    __cpuid(regs, 1);
    if ((regs[2] & (1 << 27)) == 0 || /* OSXSAVE */
        (regs[2] & (1 << 28)) == 0 || /* AVX */
        (regs[2] & (1 << 12)) == 0) { /* FMA */
        return 0;
    }
    xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0; /* AVX2 */
}

#elif (defined(__GNUC__) || defined(__clang__)) && \
      (defined(__x86_64__) || defined(__i386__))

#include <cpuid.h>
#include <stdint.h>

static uint64_t read_xcr0(void) {
    uint32_t eax;
    uint32_t edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}

int simd_avx2_fma_supported(void) {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) ||
        (ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0 ||
        (ecx & bit_FMA) == 0 || (read_xcr0() & 0x6) != 0x6) {
        return 0;
    }
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return 0;
    }
    return (ebx & bit_AVX2) != 0;
}

#else

int simd_avx2_fma_supported(void) {
    return 0;
}

#endif
