#!/usr/bin/env python3
"""Trace a square SGEMM through a small inclusive L1/L2 cache model."""

from __future__ import annotations

import argparse
import collections
import dataclasses
from typing import DefaultDict


@dataclasses.dataclass(frozen=True)
class CacheConfig:
    name: str
    size: int
    ways: int
    line_size: int = 64

    @property
    def sets(self) -> int:
        return self.size // (self.ways * self.line_size)


class Cache:
    def __init__(self, config: CacheConfig) -> None:
        if config.sets <= 0 or config.sets & (config.sets - 1):
            raise ValueError("cache set count must be a positive power of two")
        self.config = config
        self._sets = [collections.OrderedDict() for _ in range(config.sets)]

    def access(self, address: int) -> bool:
        """Return True on hit and update exact LRU state."""
        line = address // self.config.line_size
        set_index = line & (self.config.sets - 1)
        tag = line // self.config.sets
        entries = self._sets[set_index]
        if tag in entries:
            entries.move_to_end(tag)
            return True
        entries[tag] = None
        if len(entries) > self.config.ways:
            entries.popitem(last=False)
        return False


class Hierarchy:
    def __init__(self, l1: CacheConfig, l2: CacheConfig) -> None:
        self.l1 = Cache(l1)
        self.l2 = Cache(l2)
        self.accesses: DefaultDict[str, int] = collections.defaultdict(int)
        self.l1_misses: DefaultDict[str, int] = collections.defaultdict(int)
        self.l2_misses: DefaultDict[str, int] = collections.defaultdict(int)

    def access(self, label: str, address: int) -> None:
        self.accesses[label] += 1
        if self.l1.access(address):
            return
        self.l1_misses[label] += 1
        if not self.l2.access(address):
            self.l2_misses[label] += 1


def matrix_address(base: int, n: int, row: int, column: int) -> int:
    return base + (row * n + column) * 4


def simulate(n: int, order: str, l1_size: int, l2_size: int) -> Hierarchy:
    hierarchy = Hierarchy(
        CacheConfig("L1D", l1_size, ways=8),
        CacheConfig("L2", l2_size, ways=8),
    )
    # Large, non-overlapping, line-aligned virtual regions avoid accidental aliasing.
    a_base, b_base, c_base = 0x10000000, 0x20000000, 0x30000000

    if order == "ijk":
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    hierarchy.access("A", matrix_address(a_base, n, i, k))
                    hierarchy.access("B", matrix_address(b_base, n, k, j))
                hierarchy.access("C", matrix_address(c_base, n, i, j))
    elif order == "ikj":
        for i in range(n):
            for j in range(n):
                hierarchy.access("C", matrix_address(c_base, n, i, j))
            for k in range(n):
                hierarchy.access("A", matrix_address(a_base, n, i, k))
                for j in range(n):
                    hierarchy.access("B", matrix_address(b_base, n, k, j))
                    hierarchy.access("C", matrix_address(c_base, n, i, j))
    else:
        raise ValueError(f"unsupported loop order: {order}")
    return hierarchy


def total(values: dict[str, int]) -> int:
    return sum(values.values())


def print_report(name: str, hierarchy: Hierarchy) -> None:
    accesses = total(hierarchy.accesses)
    l1_misses = total(hierarchy.l1_misses)
    l2_misses = total(hierarchy.l2_misses)
    print(f"{name}")
    print(f"  accesses:  {accesses:>12,}")
    print(f"  L1 misses: {l1_misses:>12,}  ({100.0 * l1_misses / accesses:6.2f}%)")
    print(f"  L2 misses: {l2_misses:>12,}  ({100.0 * l2_misses / accesses:6.2f}%)")
    print("  per matrix:")
    for label in ("A", "B", "C"):
        print(
            f"    {label}: {hierarchy.l1_misses[label]:>10,} L1, "
            f"{hierarchy.l2_misses[label]:>10,} L2"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128, help="matrix dimension (default: 128)")
    parser.add_argument("--l1-kib", type=int, default=32, help="8-way L1D size in KiB")
    parser.add_argument("--l2-kib", type=int, default=256, help="8-way L2 size in KiB")
    args = parser.parse_args()
    if args.size <= 0 or args.l1_kib <= 0 or args.l2_kib <= 0:
        parser.error("all sizes must be positive")

    configs = {
        "naive ijk": simulate(args.size, "ijk", args.l1_kib * 1024, args.l2_kib * 1024),
        "cache-friendly ikj (scalar and AVX2)": simulate(
            args.size, "ikj", args.l1_kib * 1024, args.l2_kib * 1024
        ),
    }
    print(
        f"SGEMM cache simulation: N={args.size}, "
        f"L1D={args.l1_kib} KiB, L2={args.l2_kib} KiB, 8-way, 64-byte lines\n"
    )
    for name, hierarchy in configs.items():
        print_report(name, hierarchy)
        print()

    naive = total(configs["naive ijk"].l1_misses)
    friendly = total(configs["cache-friendly ikj (scalar and AVX2)"].l1_misses)
    print(f"ikj reduces modeled L1 misses by {100.0 * (naive - friendly) / naive:.1f}%.")
    print("Scalar ikj and AVX2 share this access pattern; their remaining gap is SIMD throughput.")


if __name__ == "__main__":
    main()
