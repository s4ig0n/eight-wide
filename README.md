# SIMD Weekend: Cache-Friendly SGEMM

A small, measurable C performance project that multiplies square, row-major
single-precision matrices (`C = A x B`) four ways:

1. **naive-ijk** — the obvious scalar triple loop; column walks through `B`
   have poor spatial locality.
2. **scalar-ikj** — the same computation with cache-friendly loop order. The
   compiler's auto-vectorizer is disabled for this file.
3. **avx2-fma-ikj** — the same `ikj` access pattern using eight-wide AVX2
   vectors and fused multiply-add instructions.
4. **avx2-blocked** — the AVX2/FMA kernel tiled in 128 x 128 x 128 blocks so
   working rows of `B` and `C` are reused before they leave the smaller caches.

This split makes three effects visible: `naive-ijk` versus `scalar-ikj`
isolates loop-order locality, `scalar-ikj` versus `avx2-fma-ikj` isolates
explicit SIMD, and `avx2-fma-ikj` versus `avx2-blocked` isolates tiling. Odd
matrix sizes are supported by a scalar tail.

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
avx2-blocked         ...         ...           ...x          ...
SIMD-only speedup (same ikj access pattern): ...x
Blocking gain over unblocked AVX2: ...x
```

Use a Release build, close noisy applications, and compare several matrix
sizes. Sizes near 64, 256, 512, and 1024 help expose transitions between cache
levels. The benchmark warms each kernel and reports the best timed run to
reduce scheduler noise; it also prints a checksum so the work remains
observable.

Floating-point accumulation order differs between `ijk`, `ikj`, and FMA, so
the results are compared with a size-scaled tolerance rather than bitwise.

### Results

Release build measured on an Intel64 Family 6 Model 154 host with Clang 20.1.2
on 2026-11-18. Each entry is the best of five runs except N=1024 (best of
three); results will vary with CPU frequency, thermals, and background load.

| N | naive-ijk | scalar-ikj | AVX2/FMA | blocked AVX2 | total speedup | blocking gain |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.26 GFLOP/s | 5.55 GFLOP/s | 24.84 GFLOP/s | 30.61 GFLOP/s | 13.53x | 1.23x |
| 512 | 1.64 GFLOP/s | 6.09 GFLOP/s | 23.48 GFLOP/s | 29.15 GFLOP/s | 17.79x | 1.24x |
| 1024 | 0.37 GFLOP/s | 2.89 GFLOP/s | 7.88 GFLOP/s | 9.82 GFLOP/s | 26.86x | 1.25x |

The 512 case is the clearest summary: loop order supplies a 3.72x gain,
explicit AVX2/FMA adds another 3.86x with the same access pattern, and tiling
adds 1.24x by reusing 128-row panels. At N=1024, the working set moves beyond
the smaller caches and the strided naive kernel degrades most sharply.

## Cache simulation

`tools/cache_sim.py` models exact LRU replacement in 8-way L1D and L2 caches,
with 64-byte cache lines and write allocation. It also exports the same abstract
matrix addresses as Valgrind-compatible traces for the separately validated
C++ [Cache Simulator](https://github.com/s4ig0n/Cache-Simulator), tying the two
projects together instead of treating the Python model as an independent
source of truth.

```sh
python tools/cache_sim.py --size 128
python -m unittest discover -s tests -p "test_*.py" -v
```

To replay both loop orders through the C++ simulator, generate smaller traces
and point `cachesim` at them (the example uses an 8 KiB L1 to make the locality
difference visible at N=64):

```sh
python tools/cache_sim.py --size 64 --l1-kib 8 --trace-dir build/traces
cachesim -s 8K -b 64 -a 8 --l2-size 256K build/traces/naive_ijk.trace
cachesim -s 8K -b 64 -a 8 --l2-size 256K build/traces/cache_friendly_ikj.trace
```

The C++ simulator reports exactly the same L1 miss counts as the Python model:

| N=64, 8 KiB L1 | accesses | L1 misses | L1 miss rate | L2 misses | AMAT |
|---|---:|---:|---:|---:|---:|
| naive `ijk` | 528,384 | 270,592 | 51.21% | 768 | 10.29 cycles |
| cache-friendly `ikj` | 794,624 | 16,896 | 2.13% | 768 | 4.35 cycles |

The `ikj` trace contains more explicit accesses because every accumulation is
represented as a load and a store to `C`; despite that, it cuts L1 misses by
93.8% and modeled AMAT by 57.7%. The C++ simulator has been validated against
Cachegrind to within 0.06%, so these cross-project results are the stronger
check on the explanatory Python model.

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

## Project layout

```text
include/simd_matmul.h   public kernel API
src/matmul_naive.c     plain ijk implementation
src/matmul_scalar.c    scalar cache-friendly implementation
src/matmul_avx2.c      AVX2/FMA implementation
src/matmul_blocked_avx2.c  cache-blocked AVX2/FMA implementation
src/benchmark.c        timing, validation, CSV output
tools/cache_sim.py     cache model and C++ simulator trace exporter
tests/                 native correctness and simulator tests
```

## License

MIT
