"""§3.1 pre-freeze guardrails for snap_v2 (annotation-protocol.md).

Three checks the protocol requires BEFORE freezing snap_v2 / opening annotation:

  overlay_snaps(name)   §3.1 #1 — overlay 30-50 snapped onsets on the envelope so a
                        human can verify each lands on the MAIN transient's foot (not
                        the precursor, not 7-10 ms interference). Focus on the low-SNR
                        files (BJ_C1 / IntlSB20) and dense passages (BJ_C3).

  sensitivity(name)     §3.1 #2 — re-snap at X = 20 vs 14 and 20 vs 26 dB backtrack
                        depth; report the PER-CLICK displacement distribution
                        (median + p95 + max, ms) + 存疑 rate — the "口径不确定度"
                        reported with eval. Also broken out for ISOLATED clicks
                        (>=120 ms from both neighbours), since pre-confirmation
                        seeds may still contain detector noise in dense spans.

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


def _seed_marks(name: str) -> tuple[list[int], list[str]]:
    """ALL rows (sorted) from labels/<name>.seed.csv — windows need every mark."""
    path = LABELS_DIR / f"{name}.seed.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["sample"]))
    return [int(r["sample"]) for r in rows], [r["type"] for r in rows]


def _load_env(name: str) -> tuple[np.ndarray, int, np.ndarray]:
    y, sr = dsp.load(WAV_DIR / f"{name}.wav")
    return y, sr, dsp.hp_rms_envelope(y, sr)


def _isolated_mask(marks: list[int], sr: int, iso_ms: float = 120.0) -> np.ndarray:
    """True where a mark is >= iso_ms from BOTH neighbours (clean calibration subset)."""
    a = np.asarray(marks, dtype=float)
    gp = np.r_[np.inf, np.diff(a)] / sr * 1000.0
    gn = np.r_[np.diff(a), np.inf] / sr * 1000.0
    return (gp >= iso_ms) & (gn >= iso_ms)


# --- §3.1 #1: visual onset overlay --------------------------------------------

def overlay_snaps(name: str, n: int = 40, start: int | None = None,
                  half_ms: float = 20.0) -> Path:
    """Overlay n snap_v2 onsets on the RMS envelope (log y), grid of subplots.

    Per panel: green/red solid = onset (red = argmin fallback, 存疑), magenta dot =
    main-transient peak, orange dotted = peak - 20 dB backtrack threshold, blue
    dashed = the seed mark. Eyeball: onset must sit at the MAIN rise's foot, with
    the precursor (if visible ~15-25 ms earlier) left OUTSIDE the cut.
    """
    y, sr, env = _load_env(name)
    marks, _types = _seed_marks(name)
    results = snap.snap_v2_marks(env, marks, sr)
    if start is None:                       # representative: even stride across the file
        idx = (np.linspace(0, len(marks) - 1, n).round().astype(int)
               if len(marks) > n else np.arange(len(marks)))
    else:                                   # contiguous span (e.g. inspect a dense burst)
        idx = np.arange(start, min(start + n, len(marks)))

    half = int(half_ms * sr / 1000.0)
    cols = 8
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.7 * rows), squeeze=False)
    n_fb = 0
    for k, i in enumerate(idx):
        r, c = results[i], marks[i]
        n_fb += r.fallback
        ax = axes[k // cols][k % cols]
        a, b = max(r.sample - half, 0), min(r.sample + 2 * half, len(env) - 1)
        t = (np.arange(a, b) - r.sample) / sr * 1000.0
        ax.semilogy(t, np.maximum(env[a:b], 1e-9), lw=0.7, color="0.3")
        ax.axhline(r.threshold, color="orange", lw=0.6, ls=":")          # peak - 20 dB
        ax.axvline(0, color="g" if not r.fallback else "r", lw=1.0)      # snapped onset
        ax.axvline((c - r.sample) / sr * 1000.0, color="b", lw=0.5, ls="--")  # seed mark
        ax.plot((r.peak - r.sample) / sr * 1000.0, r.peak_amp, "m.", ms=4)    # main peak
        ax.set(xticks=[], yticks=[], title=("FB" if r.fallback else "ok"))
        ax.title.set_fontsize(6)
    for k in range(len(idx), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{name}: {snap.SNAP_VERSION} (green=onset, m.=peak, blue--=seed, "
                 f"orange:=mid-thr) | {n_fb}/{len(idx)} doubt(prom<12dB)", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"{name}_snap_overlay.png"
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[overlay] {name}: {len(idx)} marks (from idx {start}), "
          f"{n_fb} fallback -> {out.relative_to(REPO)}")
    return out


# --- §3.1 #2: sensitivity = per-click displacement distribution ----------------

def _disp_stats(base, alt, sr) -> dict:
    """Per-click displacement stats between two same-length result lists."""
    pairs = [(b, a) for b, a in zip(base, alt) if not b.fallback and not a.fallback]
    d = np.asarray([(a.sample - b.sample) / sr * 1000.0 for b, a in pairs])
    ad = np.abs(d)
    return {
        "n_clean_both": len(d),
        "abs_median_ms": float(np.median(ad)) if len(ad) else float("nan"),
        "abs_p95_ms": float(np.percentile(ad, 95)) if len(ad) else float("nan"),
        "abs_max_ms": float(ad.max()) if len(ad) else float("nan"),
        "signed_median_ms": float(np.median(d)) if len(d) else float("nan"),
        "d": d,
    }


def sensitivity(name: str, x_base: float = 0.5, x_alts=(0.35, 0.65)) -> dict:
    """Re-snap at mid_frac=x_base vs each alt; per-click |Δonset| distribution + 存疑率.

    Only `down` marks enter the statistics (ups are best-effort), but windows are
    built from ALL marks. Also reported for the isolated (>=120 ms) subset.
    存疑 = low prominence (range < 12 dB) — independent of mid_frac, so it is
    reported once; the sweep measures pure calibration uncertainty of the foot.
    """
    y, sr, env = _load_env(name)
    marks, types = _seed_marks(name)
    is_down = np.asarray([t == "down" for t in types])
    iso = _isolated_mask(marks, sr) & is_down

    res = {x: snap.snap_v2_marks(env, marks, sr, mid_frac=x)
           for x in (x_base, *x_alts)}
    base = res[x_base]
    n_down = int(is_down.sum())
    fb = {x: sum(r.fallback for r, d in zip(res[x], is_down) if d) for x in res}
    print(f"[sensitivity] {name}: downs={n_down} (isolated={int(iso.sum())}) | "
          f"存疑(prom<12dB): {fb[x_base]} ({100*fb[x_base]/max(n_down,1):.1f}%)")

    out_stats = {"n_down": n_down, "doubt": fb}
    fig, axes = plt.subplots(1, len(x_alts), figsize=(6 * len(x_alts), 3.2), squeeze=False)
    for j, x in enumerate(x_alts):
        sub_all = _disp_stats([b for b, d in zip(base, is_down) if d],
                              [a for a, d in zip(res[x], is_down) if d], sr)
        sub_iso = _disp_stats([b for b, m in zip(base, iso) if m],
                              [a for a, m in zip(res[x], iso) if m], sr)
        out_stats[x] = {k: v for k, v in sub_all.items() if k != "d"}
        out_stats[(x, "iso")] = {k: v for k, v in sub_iso.items() if k != "d"}
        print(f"  |Δonset| mid_frac {x_base:g}->{x:g}  all-downs: "
              f"median={sub_all['abs_median_ms']:.2f} p95={sub_all['abs_p95_ms']:.2f} "
              f"max={sub_all['abs_max_ms']:.2f} ms (signed med "
              f"{sub_all['signed_median_ms']:+.2f}, n={sub_all['n_clean_both']})")
        print(f"                                isolated: "
              f"median={sub_iso['abs_median_ms']:.2f} p95={sub_iso['abs_p95_ms']:.2f} "
              f"max={sub_iso['abs_max_ms']:.2f} ms (n={sub_iso['n_clean_both']})")
        ax = axes[0][j]
        if len(sub_all["d"]):
            ax.hist(sub_all["d"], bins=60, color="steelblue")
            ax.axvline(sub_all["signed_median_ms"], color="r", lw=1, label="median")
            ax.legend(fontsize=8)
        ax.set(xlabel=f"d_onset (ms), mid_frac {x:g} minus {x_base:g}", ylabel="clicks",
               title=f"{name}: snap_v2 sensitivity  n={len(sub_all['d'])}")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_sensitivity.png"
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO)}")
    return out_stats


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
