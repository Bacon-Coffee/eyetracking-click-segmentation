"""Verification for snap_v2 + seed export (annotation-protocol.md §3 / §3.1 / §7).

Plain-assert script (no pytest dependency):  python3 tests/test_snap_v2.py
Covers: dsp dB helpers, synthetic-onset recovery, precursor robustness,
window clamping, low-prominence doubt flag, the SV-export round-trip, the min-gap
merge, and the snap_v1->v2 decision (precursor defeats v1 not v2 + seed-cleanliness
invariants — see docs/snap-v1-clean-audit.md).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dsp                      # noqa: E402
import export_seeds as es       # noqa: E402
import snap                     # noqa: E402

SR = dsp.SR


def synth_click(n: int, onset: int, f_hz: float = 6000.0, amp: float = 0.5,
                decay_ms: float = 4.0, noise: float = 1e-3,
                seed: int = 0) -> np.ndarray:
    """Noise floor + exponentially decaying HF burst starting exactly at `onset`."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, noise, n).astype(np.float64)
    t = np.arange(n - onset) / SR
    y[onset:] += amp * np.exp(-t / (decay_ms / 1000.0)) * np.sin(2 * np.pi * f_hz * t)
    return y.astype(np.float32)


def test_db_helpers():
    assert abs(dsp.db_gain(6.0) - 1.9953) < 1e-3
    assert abs(dsp.db_gain(-20.0) - 0.1) < 1e-12
    assert abs(dsp.amp_to_db_ratio(2.0, 1.0) - 6.0206) < 1e-3
    print("ok  db helpers")


def test_synthetic_onset():
    onset = SR // 2
    y = synth_click(SR, onset)
    env = dsp.hp_rms_envelope(y, SR)
    mark = onset + round(0.012 * SR)            # human mark 12 ms late — within ±20 ms
    r = snap.snap_v2_marks(env, [mark], SR)[0]
    err_ms = abs(r.sample - onset) / SR * 1000.0
    assert not r.fallback, "clean synthetic click must not be doubt"
    assert err_ms <= 1.0, f"onset error {err_ms:.2f} ms > 1 ms"
    print(f"ok  synthetic onset (|err| = {err_ms:.2f} ms, 确定)")


def test_precursor_skipped():
    """A -18 dB precursor 20 ms before the main click must NOT capture the onset."""
    onset = SR // 2
    pre = onset - round(0.020 * SR)
    y = synth_click(SR, onset)
    y += synth_click(SR, pre, amp=0.5 * dsp.db_gain(-18.0), decay_ms=2.0, noise=0.0)
    env = dsp.hp_rms_envelope(y, SR)
    r = snap.snap_v2_marks(env, [onset], SR)[0]
    err_ms = abs(r.sample - onset) / SR * 1000.0
    assert err_ms <= 1.0, f"onset pulled toward precursor (err {err_ms:.2f} ms)"
    print(f"ok  precursor skipped (|err| = {err_ms:.2f} ms)")


def test_windows_clamped():
    marks = [1000, 2000, 50_000]
    wins = snap.windows_from_marks(marks, 100_000, SR)   # ±30 ms = ±1440 samples
    assert wins[0][0] == 0 and wins[0][1] == 1500        # left clamp + midpoint
    assert wins[1][0] == 1501                            # midpoint, no overlap
    assert wins[1][1] == 2000 + 1440                     # full half-window
    assert wins[2] == (50_000 - 1440, 50_000 + 1440)
    assert all(a[1] < b[0] for a, b in zip(wins, wins[1:])), "windows overlap"
    try:
        snap.windows_from_marks([5, 3], 100, SR)
        raise AssertionError("unsorted marks must raise")
    except ValueError:
        pass
    print("ok  windows clamped, non-overlapping, sorted-checked")


def test_low_prominence_doubt():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1e-2, SR).astype(np.float32)     # noise only, no click
    env = dsp.hp_rms_envelope(y, SR)
    r = snap.snap_v2_marks(env, [SR // 2], SR)[0]
    assert r.fallback and r.confidence == "存疑", \
        f"noise-only window must be doubt (range {r.range_db:.1f} dB)"
    print(f"ok  low prominence -> 存疑 (range {r.range_db:.1f} dB)")


def test_sv_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        sv = Path(td) / "sv.csv"
        out = Path(td) / "out.csv"
        sv.write_text('12345,0.7,"down"\n19632,0.5,"up"\n25000,0.1,"x"\n',
                      encoding="utf-8")
        n = es.from_sv_export(sv, out, SR)
        rows = out.read_text(encoding="utf-8").strip().splitlines()
        assert n == 3 and rows[0] == "sample,time_s,type,confidence"
        assert rows[1].startswith("12345,0.257188,down")
        assert rows[2].startswith("19632,") and ",up," in rows[2]
        assert ",down," in rows[3]                       # unknown label -> down
    print("ok  SV export round-trip")


def test_merge_close_peaks():
    env = np.zeros(100_000, dtype=np.float32)
    peaks = np.array([10_000, 10_300, 11_000, 50_000])   # first three within 30 ms
    for p, a in zip(peaks, (0.2, 0.9, 0.3, 0.5)):
        env[p] = a
    merged = es.merge_close_peaks(peaks, env, SR)
    assert list(merged) == [10_300, 50_000], f"got {list(merged)}"
    print("ok  min-gap merge keeps strongest")


def test_precursor_defeats_v1_not_v2():
    """Locks the snap_v1->snap_v2 decision (docs/snap-v1-clean-audit.md): on a click
    with a real ~15-25 ms precursor that leaves NO quiet 1 ms foot before the main
    rise, RETIRED snap_v1 cannot arm and FALLS BACK, while peak-relative snap_v2
    lands on the main foot. This is the mechanism behind the ~55% clean isolated
    snap_v1 fallback — file-independent so it can't silently rot when seeds change."""
    onset = SR // 2
    pre = onset - round(0.015 * SR)                      # precursor 15 ms before main
    y = synth_click(SR, onset, amp=0.5, decay_ms=4.0, noise=1e-3)
    # slow-decaying precursor (-12 dB, 8 ms) fills the pre-onset gap -> no quiet foot
    y += synth_click(SR, pre, amp=0.5 * dsp.db_gain(-12.0), decay_ms=8.0, noise=0.0, seed=1)
    env = dsp.hp_rms_envelope(y, SR)
    mark = onset

    v1 = snap.snap_one(env, mark, SR)
    assert v1.fallback, "snap_v1 must FALL BACK on the precursor-laden click"

    w = snap.windows_from_marks([mark], len(env), SR)[0]
    v2 = snap.snap_v2_one(env, w, SR)
    err_ms = abs(v2.sample - onset) / SR * 1000.0
    assert not v2.fallback, "snap_v2 must NOT be doubt (clear main peak)"
    assert err_ms <= 1.0, f"snap_v2 onset error {err_ms:.2f} ms > 1 ms"
    print(f"ok  precursor defeats snap_v1 (fallback) not snap_v2 (|err|={err_ms:.2f} ms)")


def test_seed_cleanliness_invariants():
    """Locks that merge_close_peaks removes the two seed pollutions named in the
    report — exact-duplicate rows and 6-11 ms periodic-interference pairs. After
    merge, survivors are >= 30 ms apart, with no duplicate and no 6-11 ms gap."""
    ms = lambda x: round(x / 1000 * SR)
    env = np.zeros(120_000, dtype=np.float32)
    a0 = 10_000
    peaks = [a0, a0, a0 + ms(7), a0 + ms(14)]            # dup + two 7 ms members
    b0 = a0 + ms(14) + ms(35)                            # next cluster, >30 ms after A
    peaks += [b0, b0 + ms(10)]                           # 10 ms interference pair
    peaks += [b0 + ms(80)]                               # a clean well-separated peak
    peaks = np.array(sorted(peaks))
    for i, p in enumerate(peaks):
        env[p] = 0.1 + 0.01 * i                          # distinct amps -> deterministic

    merged = es.merge_close_peaks(peaks, env, SR)
    gaps = np.diff(merged)
    min_gap = round(es.MIN_PEAK_GAP_MS / 1000 * SR)
    assert len(merged) == len(np.unique(merged)), "exact duplicates survived merge"
    assert (gaps >= min_gap).all(), f"gap < {es.MIN_PEAK_GAP_MS} ms survived: {list(gaps)}"
    gaps_ms = gaps / SR * 1000.0
    assert not ((gaps_ms >= 6) & (gaps_ms <= 11)).any(), "6-11 ms interference pair survived"
    print(f"ok  seed cleanliness: {len(peaks)}->{len(merged)} peaks "
          f"(no dup / no 6-11 ms / all >= {es.MIN_PEAK_GAP_MS:g} ms)")


if __name__ == "__main__":
    test_db_helpers()
    test_synthetic_onset()
    test_precursor_skipped()
    test_windows_clamped()
    test_low_prominence_doubt()
    test_sv_roundtrip()
    test_merge_close_peaks()
    test_precursor_defeats_v1_not_v2()
    test_seed_cleanliness_invariants()
    print("\nALL PASS")
