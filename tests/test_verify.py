"""Verification for verify.py (pipeline step 4 — template-matching NCC).

Plain-assert script (no pytest dependency):  python3 tests/test_verify.py
Covers: template-positive high NCC, noise-burst low NCC, NCC bounds + amplitude
invariance, ±2 ms lag recovery, GMM bimodal threshold + non-bimodal graceful
fallback, sub-30 ms close-pair / chain dedup, and file-edge clipping.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dsp                       # noqa: E402
import verify                    # noqa: E402

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


def _train(onsets, total, **kw) -> np.ndarray:
    """Sum several identical-shape clicks (varied noise seed) into one waveform."""
    y = np.zeros(total, dtype=np.float64)
    for k, o in enumerate(onsets):
        y += synth_click(total, o, seed=k, **kw)
    return y.astype(np.float32)


def _env(y: np.ndarray) -> np.ndarray:
    """The production surface: HP2k -> 1 ms RMS energy envelope."""
    return dsp.hp_rms_envelope(y, SR)


def _template_from(onsets, total, **kw):
    """Build a verify template (envelope domain) from clicks at `onsets`.

    Returns (env, tmpl). Envelope, not bipolar waveform — matches production
    (verify._score_file): clicks share an energy-shape, not a carrier phase.
    """
    env = _env(_train(onsets, total, **kw))
    tmpl, n = verify.build_template(env, onsets)
    assert tmpl is not None and n == len(onsets)
    return env, tmpl


def test_template_positive_high_ncc():
    total = SR
    train = [round(0.08 * SR) * k + round(0.05 * SR) for k in range(8)]
    env, tmpl = _template_from(train, total)
    # held-out click of the SAME shape at a fresh onset -> high NCC
    held = round(0.9 * SR)
    envh = _env(synth_click(total, held, seed=99))
    score = verify.ncc_score(envh, held, tmpl)
    assert score > 0.9, f"template-shaped click scored low: {score:.3f}"
    print(f"ok  template positive high NCC (score={score:.3f})")


def test_noise_burst_low_ncc():
    total = SR
    train = [round(0.08 * SR) * k + round(0.05 * SR) for k in range(8)]
    env, tmpl = _template_from(train, total)
    # (a) noise-only window -> ~flat envelope, unlike a peaked click
    rng = np.random.default_rng(7)
    envn = _env(rng.normal(0.0, 1e-2, total).astype(np.float32))
    s_noise = verify.ncc_score(envn, total // 2, tmpl)
    # (b) broadband noise BURST: sustained ~10 ms plateau (no click rise/decay shape)
    yb = rng.normal(0.0, 1e-3, total).astype(np.float64)
    burst = total // 2
    yb[burst:burst + round(0.010 * SR)] += rng.normal(0.0, 0.5, round(0.010 * SR))
    envb = _env(yb.astype(np.float32))
    s_burst = verify.ncc_score(envb, burst, tmpl)
    assert s_noise < 0.5, f"noise-only scored high: {s_noise:.3f}"
    assert s_burst < 0.8, f"noise burst not separable from a click: {s_burst:.3f}"
    print(f"ok  noise low NCC (noise={s_noise:.3f}, burst={s_burst:.3f})")


def test_ncc_bounds_and_amplitude_invariance():
    total = SR
    train = [round(0.08 * SR) * k + round(0.05 * SR) for k in range(8)]
    env, tmpl = _template_from(train, total)
    held = round(0.9 * SR)
    envh = _env(synth_click(total, held, seed=42))
    s_loud = verify.ncc_score(envh, held, tmpl)
    s_quiet = verify.ncc_score(0.1 * envh, held, tmpl)     # whole surface scaled
    assert abs(s_loud - s_quiet) < 1e-6, \
        f"L2 not amplitude-invariant: {s_loud:.6f} vs {s_quiet:.6f}"
    # every score must be within [-1, 1]
    for o in train + [held]:
        s = verify.ncc_score(env, o, tmpl)
        assert -1.0 - 1e-9 <= s <= 1.0 + 1e-9, f"NCC out of [-1,1]: {s}"
    print(f"ok  NCC bounds + amplitude invariance (Δ={abs(s_loud - s_quiet):.2e})")


def test_lag_search_recovers_offset():
    total = SR
    train = [round(0.08 * SR) * k + round(0.05 * SR) for k in range(8)]
    env, tmpl = _template_from(train, total)
    onset = round(0.9 * SR)
    envh = _env(synth_click(total, onset, seed=5))
    s_on = verify.ncc_score(envh, onset, tmpl)
    s_in = verify.ncc_score(envh, onset + round(0.0015 * SR), tmpl)   # +1.5 ms (within ±2)
    s_out = verify.ncc_score(envh, onset + round(0.005 * SR), tmpl)   # +5 ms (outside ±2)
    assert s_in > 0.9, f"±2 ms lag failed to recover +1.5 ms offset: {s_in:.3f}"
    assert s_out < s_in, f"+5 ms offset not penalized: in={s_in:.3f} out={s_out:.3f}"
    print(f"ok  lag search (on={s_on:.3f}, +1.5ms={s_in:.3f}, +5ms={s_out:.3f})")


def test_threshold_bimodal_and_fallback():
    rng = np.random.default_rng(0)
    hi = np.clip(rng.normal(0.95, 0.02, 250), -1, 1)
    lo = np.clip(rng.normal(0.30, 0.08, 250), -1, 1)
    bimodal = np.concatenate([hi, lo])
    thr = verify.derive_thresholds(bimodal)
    assert thr["bimodal"], f"clean bimodal not detected (sep={thr['separation']:.2f})"
    assert thr["theta_lo"] < thr["theta"] < thr["theta_hi"], \
        f"bad band order: {thr['theta_lo']:.3f}/{thr['theta']:.3f}/{thr['theta_hi']:.3f}"
    assert 0.30 < thr["theta"] < 0.95, f"θ not in the valley: {thr['theta']:.3f}"

    uni = np.clip(rng.normal(0.95, 0.02, 400), -1, 1)
    thr2 = verify.derive_thresholds(uni)
    assert not thr2["bimodal"], "unimodal wrongly flagged bimodal"
    assert thr2["basis"].startswith("fallback"), f"expected fallback, got {thr2['basis']}"
    assert thr2["theta_lo"] == thr2["theta_lo"], "fallback θ_lo must not be NaN"
    assert thr2["theta_lo"] == verify.THETA_FALLBACK_LO
    print(f"ok  threshold bimodal (θ={thr['theta']:.3f}) + unimodal fallback "
          f"({thr2['basis']})")


def test_dedup_close_pairs_and_chain():
    ms = lambda x: round(x / 1000 * SR)
    theta_hi, theta_lo = 0.7, 0.3

    def mk(sample, ncc):
        d = {"sample": sample, "ncc": ncc, "range_db": 20.0}
        d["decision"], d["reason"] = verify.classify(ncc, 20.0, theta_lo, theta_hi)
        return d

    downs = [
        mk(100_000, 0.9), mk(100_000 + ms(20), 0.1),     # (a) pair: one high one low
        mk(200_000, 0.9), mk(200_000 + ms(15), 0.9),     # (b) pair: both high <30 ms
        mk(300_000, 0.9),                                 # (c) lone high, well separated
        mk(400_000, 0.9), mk(400_000 + ms(10), 0.1),     # (d) 3-chain: one high two low
        mk(400_000 + ms(20), 0.1),
    ]
    kept, dropped, adjudicate = verify.dedup_close_downs(downs, SR, theta_hi, theta_lo)
    k = {d["sample"] for d in kept}
    dr = {d["sample"] for d in dropped}
    aj = {d["sample"] for d in adjudicate}

    assert k == {100_000, 300_000, 400_000}, f"kept wrong: {sorted(k)}"
    assert dr == {100_000 + ms(20), 400_000 + ms(10), 400_000 + ms(20)}, \
        f"dropped wrong: {sorted(dr)}"
    assert aj == {200_000, 200_000 + ms(15)}, f"adjudicate wrong: {sorted(aj)}"
    reasons = {d["sample"]: d["reason"] for d in adjudicate}
    assert all(r == "both_high_too_close" for r in reasons.values()), reasons
    print(f"ok  dedup close pairs + chain (kept={len(k)} drop={len(dr)} adj={len(aj)})")


def test_edge_onset_clipping():
    total = SR
    train = [round(0.08 * SR) * k + round(0.05 * SR) for k in range(8)]
    env, tmpl = _template_from(train, total)
    seg = verify.extract_segment(env, 5)                 # onset 5 -> left edge clipped
    assert len(seg) == verify.SEG_LEN, f"clipped segment wrong length: {len(seg)}"
    s = verify.ncc_score(env, 5, tmpl)                   # must not crash; finite
    assert -1.0 - 1e-9 <= s <= 1.0 + 1e-9, f"edge NCC out of range: {s}"
    print(f"ok  edge onset clipping (len={len(seg)}, score={s:.3f})")


if __name__ == "__main__":
    test_template_positive_high_ncc()
    test_noise_burst_low_ncc()
    test_ncc_bounds_and_amplitude_invariance()
    test_lag_search_recovers_offset()
    test_threshold_bimodal_and_fallback()
    test_dedup_close_pairs_and_chain()
    test_edge_onset_clipping()
    print("\nALL PASS")
