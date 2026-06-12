"""Pure-assertion tests for src/cut.py (no audio files needed). Run: python tests/test_cut.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cut import segment_bounds  # noqa: E402


def test_bounds_partition():
    """N cuts -> N+1 contiguous, gapless, exhaustive [start, end) bounds."""
    n = 1000
    bounds = segment_bounds([100, 400, 999], n)
    assert bounds == [(0, 100), (100, 400), (400, 999), (999, 1000)]
    assert sum(b - a for a, b in bounds) == n
    for (_, e1), (s2, _) in zip(bounds, bounds[1:]):
        assert e1 == s2


def test_bounds_no_cuts():
    assert segment_bounds([], 50) == [(0, 50)]


def test_bounds_rejects_unsorted_and_duplicate():
    for bad in ([400, 100], [100, 100]):
        try:
            segment_bounds(bad, 1000)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} accepted")


def test_bounds_rejects_out_of_range():
    for bad in ([0], [1000], [1500]):
        try:
            segment_bounds(bad, 1000)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} accepted")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"  ok {fn.__name__}")
    print("all cut tests passed")
