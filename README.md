# SIMD Weekend: Cache-Friendly SGEMM

A small, measurable C performance project that multiplies square, row-major
single-precision matrices (`C = A x B`) three ways:

1. **naive-ijk** — the obvious scalar triple loop; column walks through `B`
   have poor spatial locality.
2. **scalar-ikj** — the same computation with cache-friendly loop order. The
   compiler's auto-vectorizer is disabled for this file.
3. **avx2-fma-ikj** — the same `ikj` access pattern using eight-wide AVX2
   vectors and fused multiply-add instructions.

This split makes two effects visible: `naive-ijk` versus `scalar-ikj` isolates
cache locality, while `scalar-ikj` versus `avx2-fma-ikj` isolates explicit SIMD.
Odd matrix sizes are supported by a scalar tail.

## Build and test

Requirements: an x86-64 CPU, CMake 3.20+, and MSVC, GCC, or Clang.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The benchmark checks AVX2, FMA, and OS vector-state support at runtime. On an
unsupported machine it still runs both scalar kernels safely.

On Windows with Visual Studio, the executable is normally under
`build/Release/`; with a single-config generator it is directly under `build/`.

## Benchmark

```sh
./build/simd_bench --size 512 --iterations 5
./build/simd_bench --size 512 --iterations 5 --csv results.csv
```

Example output format (numbers depend on the machine):

```text
SGEMM: C = A x B, N=512, best of 5 timed runs
kernel             time-ms    GFLOP/s     vs-naive    max-error
naive-ijk           ...         ...          1.00x            0
scalar-ikj          ...         ...           ...x          ...
avx2-fma-ikj        ...         ...           ...x          ...
SIMD-only speedup (same ikj access pattern): ...x
```

Use a Release build, close noisy applications, and compare several matrix
sizes. Sizes near 64, 256, 512, and 1024 help expose transitions between cache
levels. The benchmark warms each kernel and reports the best timed run to
reduce scheduler noise; it also prints a checksum so the work remains
observable.

Floating-point accumulation order differs between `ijk`, `ikj`, and FMA, so
the results are compared with a size-scaled tolerance rather than bitwise.

### Measured result

Release build measured on an Intel64 Family 6 Model 154 host with Clang 20.1.2
on 2026-11-18. Each entry is the best of five runs except N=1024 (best of
three); results will vary with CPU frequency, thermals, and background load.

| N | naive-ijk | scalar-ikj | AVX2/FMA | total speedup | SIMD-only speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.26 GFLOP/s | 5.88 GFLOP/s | 21.38 GFLOP/s | 9.47x | 3.64x |
| 512 | 1.60 GFLOP/s | 6.10 GFLOP/s | 24.26 GFLOP/s | 15.12x | 3.98x |
| 1024 | 0.43 GFLOP/s | 2.73 GFLOP/s | 9.22 GFLOP/s | 21.55x | 3.38x |

The 512 case is the clearest summary: loop order alone supplies a 3.80x gain,
then explicit AVX2/FMA adds another 3.98x with the same memory access pattern.
At N=1024, all implementations slow as the working set moves beyond the
smaller caches, but the strided naive kernel degrades much more sharply.

## Cache simulation

`tools/cache_sim.py` models exact LRU replacement in 8-way L1D and L2 caches,
with 64-byte cache lines and write allocation. It traces abstract addresses for
all three matrices; it is deliberately small and transparent rather than a
cycle-accurate CPU model.

```sh
python tools/cache_sim.py --size 128
python -m unittest discover -s tests -p "test_*.py" -v
```

The expected result is far fewer L1 misses for `ikj`, especially for `B`:
`ijk` steps down a column with an `N * 4` byte stride, while `ikj` streams along
contiguous rows. The scalar and AVX2 `ikj` kernels have essentially the same
cache-line trace, so cache behavior explains their common advantage over the
naive version but not the speed difference between them. That remaining gap
comes from doing eight multiply-adds per vector instruction, subject to load,
store, dependency, and frequency limits.

For N=128 with the default 32 KiB L1D and 256 KiB L2 model, the measured trace
contains 2,130,960 L1 misses for `ijk` versus 133,120 for `ikj`: a 93.8%
reduction. L2 misses are identical (3,072) because all three 64 KiB matrices
fit in the modeled L2; loop order changes how efficiently those lines are
reused in L1, not how many unique lines must be fetched from L2.

The simulator omits prefetching, TLBs, non-inclusive cache policies, hardware
replacement heuristics, and instruction-cache effects. Treat it as an
explanatory model, not a substitute for hardware counters. On Linux, a useful
follow-up measurement is:

```sh
perf stat -e cycles,instructions,cache-references,cache-misses \
  ./build/simd_bench --size 512 --iterations 5
```

## Weekend experiment plan

- **Friday evening:** build, run correctness tests, and establish naive timings.
- **Saturday:** compare loop orders, run the cache simulator, and record several
  sizes in CSV.
- **Sunday:** inspect the AVX2 kernel, measure SIMD-only speedup, try a block size
  or transposed `B`, and write down where scaling changes.

## Project layout

```text
include/simd_matmul.h   public kernel API
src/matmul_naive.c     plain ijk implementation
src/matmul_scalar.c    scalar cache-friendly implementation
src/matmul_avx2.c      AVX2/FMA implementation
src/benchmark.c        timing, validation, CSV output
tools/cache_sim.py     two-level set-associative cache model
tests/                 native correctness and simulator tests
```

## License

MIT
