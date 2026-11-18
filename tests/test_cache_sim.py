import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from cache_sim import Cache, CacheConfig, simulate, total


class CacheTests(unittest.TestCase):
    def test_lru_eviction(self):
        cache = Cache(CacheConfig("tiny", size=128, ways=2, line_size=64))
        self.assertFalse(cache.access(0))
        self.assertTrue(cache.access(0))
        self.assertFalse(cache.access(64))
        self.assertFalse(cache.access(128))
        self.assertFalse(cache.access(0))

    def test_ikj_improves_l1_locality(self):
        # A power-of-two stride exposes the same L1 set conflicts as N=128
        # with a 32 KiB cache, but keeps the unit test quick.
        naive = simulate(32, "ijk", 2 * 1024, 16 * 1024)
        friendly = simulate(32, "ikj", 2 * 1024, 16 * 1024)
        self.assertLess(total(friendly.l1_misses), total(naive.l1_misses))
        self.assertLess(friendly.l1_misses["B"], naive.l1_misses["B"])

    def test_writes_valgrind_compatible_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tiny.trace"
            simulate(2, "ijk", 1024, 4096, path)
            lines = path.read_text(encoding="ascii").splitlines()
        self.assertTrue(lines[0].startswith(" L "))
        self.assertIn(",4", lines[0])
        self.assertTrue(any(line.startswith(" S ") for line in lines))


if __name__ == "__main__":
    unittest.main()
