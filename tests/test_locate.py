"""Synthetic tests for src/locate.py window localization.

The decisive case: inside one coarse window a CLICK and a LOUDER speech-like transient
both appear; the template NCC must pick the click, not the louder hump. Also covers the
no-template fallback (louder wins — documents WHY the template is needed), a silent
window (flagged 存疑/low_prom), and the multi-candidate flag.

Run: python tests/test_locate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import locate  # noqa: E402
import snap    # noqa: E402
import verify  # noqa: E402

SR = 48000


def add_click(env, pos, amp, sr=SR, attack_ms=1.0, decay_ms=6.0):
    """Sharp rise (attack) then exponential decay — the click energy shape. Foot at pos."""
    a = max(1, int(attack_ms * sr / 1000))
    tau = decay_ms * sr / 1000
    d = int(tau * 4)
    kern = np.concatenate([np.linspace(0, amp, a, endpoint=False),
                           amp * np.exp(-np.arange(d) / tau)])
    end = min(pos + len(kern), len(env))
    env[pos:end] = np.maximum(env[pos:end], kern[:end - pos])
    return pos


def add_burst(env, pos, amp, sr=SR, attack_ms=1.0, decay_ms=50.0):
    """Sharp onset (so it is prominent and competes on snap_v2 range_db, which scales
    with loudness off a sharp foot) but a LONG slow decay — a plosive/voiced burst that
    is NOT the click energy shape. Foot at pos."""
    a = max(1, int(attack_ms * sr / 1000))
    d = int(decay_ms * sr / 1000)
    kern = np.concatenate([np.linspace(0, amp, a, endpoint=False),
                           np.linspace(amp, amp * 0.2, d)])   # still high at +15 ms
    end = min(pos + len(kern), len(env))
    env[pos:end] = np.maximum(env[pos:end], kern[:end - pos])
    return pos


def click_template(amp=0.15, sr=SR):
    """Build a click template the same way the batch does, from one clean click."""
    env = np.full(sr, 1e-4, dtype=np.float32)
    pos = sr // 2
    add_click(env, pos, amp, sr)
    half = round(snap.WIN_HALF_MS * sr / 1000.0)
    res = snap.snap_v2_one(env, (pos - half, pos + half), sr)
    seg = verify.l2_normalize(verify.extract_segment(env, int(res.sample)))
    tmpl, n = locate.build_global_template([[seg]])
    assert tmpl is not None and n == 1
    return tmpl


def _click_speech_env(sr=SR):
    """2 s env: a click at 0.5 s (amp 0.15) and a LOUDER sharp burst at 1.0 s (amp 0.40)."""
    env = np.full(2 * sr, 1e-4, dtype=np.float32)
    click_pos = add_click(env, sr // 2, 0.15, sr)            # 0.5 s
    speech_pos = add_burst(env, sr, 0.40, sr)                # 1.0 s, louder + prominent
    return env, click_pos, speech_pos


def test_template_picks_click_over_louder_speech():
    env, click_pos, speech_pos = _click_speech_env()
    tmpl = click_template()
    lo, hi = int(0.2 * SR), int(1.3 * SR)
    pick = locate.pick_click(env, SR, lo, hi, np.array([], dtype=int), tmpl)
    assert pick.winner is not None
    assert pick.confidence == "确定", (pick.confidence, pick.reasons)
    assert abs(pick.winner.onset - click_pos) < int(0.005 * SR), pick.winner.onset
    assert abs(pick.winner.onset - speech_pos) > int(0.05 * SR)


def test_no_template_louder_speech_wins():
    """Without the template, selection is prominence-only -> the louder hump wins.
    This is the failure mode the template exists to prevent."""
    env, click_pos, speech_pos = _click_speech_env()
    lo, hi = int(0.2 * SR), int(1.3 * SR)
    pick = locate.pick_click(env, SR, lo, hi, np.array([], dtype=int), tmpl=None)
    assert pick.winner is not None
    assert abs(pick.winner.onset - speech_pos) < int(0.05 * SR), pick.winner.onset


def test_silent_window_flagged():
    env = np.full(SR, 1e-4, dtype=np.float32)               # flat / silent
    tmpl = click_template()
    pick = locate.pick_click(env, SR, 0, SR - 1, np.array([], dtype=int), tmpl)
    assert pick.confidence == "存疑"
    assert "low_prom" in pick.reasons, pick.reasons


def test_multi_candidate_flag():
    env = np.full(2 * SR, 1e-4, dtype=np.float32)
    p1 = add_click(env, SR // 2, 0.15)                       # 0.5 s
    p2 = add_click(env, SR // 2 + int(0.2 * SR), 0.15)       # 0.7 s (200 ms apart)
    tmpl = click_template()
    pick = locate.pick_click(env, SR, int(0.2 * SR), int(1.0 * SR),
                             np.array([], dtype=int), tmpl)
    assert pick.winner is not None and pick.confidence == "确定"
    assert "multi_candidate" in pick.reasons, pick.reasons
    assert min(abs(pick.winner.onset - p1), abs(pick.winner.onset - p2)) < int(0.005 * SR)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"  ok {fn.__name__}")
    print("all locate tests passed")
