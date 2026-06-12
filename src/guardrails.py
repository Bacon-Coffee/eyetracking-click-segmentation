"""§3.1 pre-freeze guardrails for snap_v1 (annotation-protocol.md).

Three checks the protocol requires BEFORE freezing snap_v1 / opening annotation:

  overlay_snaps(name)   §3.1 #1 — overlay 30-50 snapped onsets on the waveform/
                        envelope so a human can verify each lands on the true "foot"
                        and none snap onto 7-10 ms periodic interference. Focus on the
                        low-SNR files (BJ_C1) and the clean reference (BJ_C3).

  sensitivity(name)     §3.1 #2 — re-snap at +6 dB vs +10 dB and report the PER-CLICK
                        displacement distribution (median + p95 + max, ms) — the
                        "口径不确定度" reported with eval. Also reports the fallback
                        (存疑) rate, which is itself a freeze gate.

  calibrate_margin(name)§5.1 — sweep the up/down margin on BJ_C3, report down/up/存疑
                        counts vs margin, to calibrate the 6 dB threshold.

Outputs: printed tables + PNGs in figures/. Read-only w.r.t. audio/labels.

Usage:
    python src/guardrails.py                 # all three checks on BJ_C3 + BJ_C1
    python src/guardrails.py sensitivity BJ_C1
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dsp
import export_seeds as es
import snap

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "data" / "wav"
LABELS_DIR = REPO / "labels"
FIG_DIR = REPO / "figures"


def _seed_downs(name: str) -> list[int]:
    """down-row samples from labels/<name>.seed.csv (coarse onsets)."""
    path = LABELS_DIR / f"{name}.seed.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return [int(r["sample"]) for r in csv.DictReader(f) if r["type"] == "down"]


def _load_env(name: str) -> tuple[np.ndarray, int, np.ndarray]:
    y, sr = dsp.load(WAV_DIR / f"{name}.wav")
    return y, sr, dsp.hp_rms_envelope(y, sr)


# --- §3.1 #1: visual onset overlay --------------------------------------------

def overlay_snaps(name: str, n: int = 40, start: int | None = None,
                  half_ms: float = 15.0) -> Path:
    """Overlay n snapped onsets on the RMS envelope (±half_ms each), grid of subplots."""
    y, sr, env = _load_env(name)
    downs = _seed_downs(name)
    if start is None:                       # representative: even stride across the file
        if len(downs) > n:
            idx = np.linspace(0, len(downs) - 1, n).round().astype(int)
            picks = [downs[i] for i in idx]
        else:
            picks = downs
    else:                                   # contiguous span (e.g. inspect a dense burst)
        picks = downs[start:start + n]

    half = int(half_ms * sr / 1000.0)
    cols = 8
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.7 * rows), squeeze=False)
    n_fb = 0
    for k, c in enumerate(picks):
        r = snap.snap_one(env, c, sr)
        n_fb += r.fallback
        ax = axes[k // cols][k % cols]
        a, b = max(r.sample - half, 0), min(r.sample + half, len(env) - 1)
        t = (np.arange(a, b) - r.sample) / sr * 1000.0
        ax.plot(t, env[a:b], lw=0.7, color="0.3")
        ax.axhline(r.threshold, color="orange", lw=0.6, ls=":")          # +6 dB threshold
        ax.axvline(0, color="g" if not r.fallback else "r", lw=1.0)      # snapped onset
        ax.axvline((c - r.sample) / sr * 1000.0, color="b", lw=0.5, ls="--")  # coarse seed
        ax.set(xticks=[], yticks=[], title=("FB" if r.fallback else "ok"))
        ax.title.set_fontsize(6)
    for k in range(len(picks), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{name}: snap_v1 onsets (green=onset, blue--=seed, orange:=+6dB) "
                 f"| {n_fb}/{len(picks)} fallback", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"{name}_snap_overlay.png"
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[overlay] {name}: {len(picks)} clicks (from idx {start}), "
          f"{n_fb} fallback -> {out.relative_to(REPO)}")
    return out


# --- §3.1 #2: sensitivity = per-click displacement distribution ----------------

def sensitivity(name: str, n: int | None = None) -> dict:
    """Re-snap at +6 vs +10 dB; report per-click |Δonset| distribution (median/p95/max)."""
    y, sr, env = _load_env(name)
    downs = _seed_downs(name)
    if n:
        downs = downs[:n]

    disp_ms, fb6, fb10 = [], 0, 0
    for c in downs:
        r6 = snap.snap_one(env, c, sr, thresh_db=6.0)
        r10 = snap.snap_one(env, c, sr, thresh_db=10.0)
        fb6 += r6.fallback
        fb10 += r10.fallback
        if not r6.fallback and not r10.fallback:      # displacement only for clean both
            disp_ms.append((r10.sample - r6.sample) / sr * 1000.0)
    d = np.asarray(disp_ms)
    ad = np.abs(d)
    stats = {
        "n": len(downs), "n_clean_both": len(d),
        "fallback_6dB": fb6, "fallback_10dB": fb10,
        "abs_median_ms": float(np.median(ad)) if len(ad) else float("nan"),
        "abs_p95_ms": float(np.percentile(ad, 95)) if len(ad) else float("nan"),
        "abs_max_ms": float(ad.max()) if len(ad) else float("nan"),
        "signed_median_ms": float(np.median(d)) if len(d) else float("nan"),
    }

    print(f"[sensitivity] {name}: clicks={stats['n']} "
          f"clean(6&10)={stats['n_clean_both']} "
          f"fallback@6dB={fb6} ({100*fb6/max(stats['n'],1):.0f}%) "
          f"fallback@10dB={fb10}")
    print(f"  |Δonset| 6->10 dB: median={stats['abs_median_ms']:.2f} "
          f"p95={stats['abs_p95_ms']:.2f} max={stats['abs_max_ms']:.2f} ms "
          f"(signed median={stats['signed_median_ms']:+.2f}; +10 dB later = positive)")

    if len(d):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(d, bins=60, color="steelblue")
        ax.axvline(stats["signed_median_ms"], color="r", lw=1, label="median")
        ax.set(xlabel="d_onset (ms), +10 dB minus +6 dB", ylabel="clicks",
               title=f"{name}: snap_v1 sensitivity (calibration uncertainty)  n={len(d)}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"{name}_sensitivity.png"
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  -> {out.relative_to(REPO)}")
    return stats


# --- §5.1: down/up margin calibration -----------------------------------------

def calibrate_margin(name: str = "BJ_C3", db_grid=(4, 5, 6, 7, 8)) -> dict:
    """Sweep the up/down margin; report #down/#up/#存疑 vs margin (calibrates §5.1)."""
    y, sr = dsp.load(WAV_DIR / f"{name}.wav")
    peaks, env_rms, _ = es.detect_candidates(y, sr)

    counts = {"db": list(db_grid), "down": [], "up": [], "doubt": []}
    print(f"[calibrate_margin] {name}:")
    print(f"  {'margin_dB':>9s}{'down':>7s}{'up':>6s}{'存疑':>6s}")
    for db in db_grid:
        rows = es.assign_down_up(peaks, env_rms, sr, up_margin_db=db,
                                 doubt_lo_db=min(es.DOUBT_LO_DB, db - 1))
        nd = sum(1 for _, t, _ in rows if t == "down")
        nu = sum(1 for _, t, _ in rows if t == "up")
        ndb = sum(1 for _, _, conf in rows if conf == "存疑")
        counts["down"].append(nd)
        counts["up"].append(nu)
        counts["doubt"].append(ndb)
        print(f"  {db:9g}{nd:7d}{nu:6d}{ndb:6d}")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(counts["db"], counts["up"], "o-", label="up")
    ax.plot(counts["db"], counts["doubt"], "s--", label="doubt")
    ax.axvline(6, color="0.6", lw=0.8, ls=":")
    ax.set(xlabel="up margin (dB weaker than pending down)", ylabel="count",
           title=f"{name}: §5.1 down/up margin calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"{name}_margin_calibration.png"
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO)}")
    return counts


def main(argv: list[str]) -> None:
    if argv:
        cmd, *rest = argv
        name = rest[0] if rest else "BJ_C3"
        {"overlay": overlay_snaps, "sensitivity": sensitivity,
         "margin": calibrate_margin}[cmd](name)
        return
    for nm in ("BJ_C3", "BJ_C1"):
        overlay_snaps(nm)
        sensitivity(nm)
    calibrate_margin("BJ_C3")


if __name__ == "__main__":
    main(sys.argv[1:])
