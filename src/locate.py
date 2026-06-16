"""locate.py — refine a coarse spreadsheet time window to a sample-accurate click onset.

New-batch core step. The xlsx (see `sheet.py`) gives a ~±1 s window per click; this
module finds the ONE real click onset inside it. The window is far too wide for
snap_v2 alone (its search half-window is ±30 ms) and these are Q&A recordings, so a
window can contain speech transients (fricatives/plosives) that are LOUDER than the
click. We therefore:

  1. enumerate candidate transients in the (margin-expanded) window — reusing the
     pipeline detector (`export_seeds.detect_candidates`, periodic-train rejected) plus
     a window-local peak pass so a click quieter than the file max is not missed;
  2. snap each candidate to its onset foot with the FROZEN snap_v2 rule
     (`snap.snap_v2_one`) — giving onset sample + pre-peak prominence (range_db);
  3. score each candidate by envelope-domain template NCC (`verify.ncc_score`) against
     a click template the BATCH builds from itself (median of prominent first-pass
     picks) — this is what tells a click apart from a louder speech transient, since
     the click's ~5–15 ms energy shape is stable while speech is not;
  4. pick the click-like + prominent candidate; flag (never silently guess) when
     nothing is click-like, the best is weak, or two strong clicks compete.

Everything operates on the canonical HP2k→1 ms RMS envelope (`dsp.hp_rms_envelope`,
returned by `detect_candidates`). snap_v2 / verify parameters are reused untouched.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

import dsp
import export_seeds as es
import snap
import verify

REPO = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO / "labels"
FIG_DIR = REPO / "figures"
SR = dsp.SR

# --- localization parameters --------------------------------------------------
MARGIN_MS = 750.0            # expand each side of the coarse window (human timing slack)
NCC_ACCEPT = 0.60            # click-like NCC gate for auto-accept (确定); cf. verify self-NCC
MIN_PROM_DB = snap.MIN_PROMINENCE_DB           # 12 dB — same prominence gate as snap/verify
CAND_MIN_DIST_MS = 15.0      # min spacing between window-local candidate peaks
CAND_HEIGHT_FRAC = 0.18      # window-local peak height floor, fraction of the window max
DEDUP_MS = 8.0               # merge candidates closer than this (keep the stronger)
MULTI_GAP_MS = 60.0          # (no-template fallback only) two prominent candidates this far apart
NCC_COMPETE_DELTA = 0.12     # candidates within this NCC of the best are "co-confident clicks"
                             # (peers) — a 0.99 winner vs a 0.65 runner-up are NOT peers
DOWNUP_MAX_MS = 150.0        # press->release (down->up) max gap (protocol §5.1 GAP_HI). A peer
                             # within this of the press is its own release, NOT a separate click,
                             # so it does not flag ambiguity; a peer beyond it is a 2nd click.
# Quality bar for an AUTO-CONFIDENT (确定) pick. Above the eligibility floor (0.60 / 12 dB):
# a located onset whose template match or prominence is weaker than this is the most
# error-prone (often a speech swell on a low-SNR file), so it is downgraded to 存疑 +
# flagged low_conf for a human look rather than silently auto-accepted.
QUALITY_NCC = 0.90
QUALITY_DB = 18.0

_HALF = round(snap.WIN_HALF_MS * SR / 1000.0)  # snap_v2 backtrack-window half-width


@dataclass(frozen=True)
class Candidate:
    onset: int          # snapped onset sample (the cut point if chosen)
    peak: int           # main-transient peak sample
    peak_amp: float     # envelope amplitude at the peak
    range_db: float     # snap_v2 pre-peak prominence
    ncc: float          # template NCC (nan if no template)


@dataclass
class Pick:
    winner: Candidate | None
    confidence: str                       # 确定 | 存疑
    reasons: tuple[str, ...] = ()
    candidates: list[Candidate] = field(default_factory=list)


def window_bounds(t_lo: float | None, t_hi: float | None, sr: int, n: int,
                  margin_ms: float = MARGIN_MS) -> tuple[int, int]:
    """Coarse [t_lo, t_hi] seconds -> margin-expanded sample bounds, clamped to [0, n-1].

    `t_lo is None` means "the whole clip" (a QN separate-audio clip is treated as one
    click in isolation)."""
    if t_lo is None:
        return 0, n - 1
    lo = max(int(round((t_lo - margin_ms / 1000.0) * sr)), 0)
    hi = min(int(round((t_hi + margin_ms / 1000.0) * sr)), n - 1)
    if hi <= lo:
        hi = min(lo + 1, n - 1)
    return lo, hi


def window_candidates(env: np.ndarray, sr: int, lo: int, hi: int,
                      global_peaks: np.ndarray) -> list[int]:
    """Candidate transient peaks inside [lo, hi]: pipeline detector peaks in the window
    UNION a window-local peak pass (catches clicks below the global 0.12*max floor),
    deduped within DEDUP_MS keeping the louder."""
    cands = {int(p) for p in global_peaks if lo <= p <= hi}
    seg = env[lo:hi + 1]
    if len(seg):
        thr = CAND_HEIGHT_FRAC * float(seg.max())
        dist = max(1, round(CAND_MIN_DIST_MS * sr / 1000.0))
        pk, _ = sps.find_peaks(seg, height=thr, distance=dist)
        cands.update(lo + int(i) for i in pk)
        cands.add(lo + int(np.argmax(seg)))            # always keep the loudest sample
    if not cands:
        return []
    dd = max(1, round(DEDUP_MS * sr / 1000.0))
    merged: list[int] = []
    for p in sorted(cands):
        if merged and p - merged[-1] < dd:
            if env[p] > env[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def score_candidates(env: np.ndarray, sr: int, cands: list[int],
                     tmpl: np.ndarray | None) -> list[Candidate]:
    """Snap each candidate to its onset (snap_v2) and score it by template NCC."""
    out: list[Candidate] = []
    for p in cands:
        res = snap.snap_v2_one(env, (max(p - _HALF, 0), int(p)), sr)
        ncc = verify.ncc_score(env, res.sample, tmpl) if tmpl is not None else float("nan")
        out.append(Candidate(int(res.sample), int(res.peak), float(res.peak_amp),
                             float(res.range_db), float(ncc)))
    return out


def pick_click(env: np.ndarray, sr: int, lo: int, hi: int,
               global_peaks: np.ndarray, tmpl: np.ndarray | None) -> Pick:
    """Choose the click onset inside [lo, hi]. Pure (no IO) — the testable core.

    A candidate is ELIGIBLE if it is click-like (NCC >= NCC_ACCEPT, when a template
    exists) AND prominent (range_db >= MIN_PROM_DB). Among eligibles pick the highest
    NCC -> 确定; two eligibles > MULTI_GAP_MS apart -> still pick, flag multi_candidate.
    If none eligible, fall back to the best-effort candidate -> 存疑 with the failing
    reason(s) so a human adjudicates."""
    cands = score_candidates(env, sr, window_candidates(env, sr, lo, hi, global_peaks), tmpl)
    if not cands:
        return Pick(None, "存疑", ("no_candidate",), [])

    has_tmpl = tmpl is not None
    key = (lambda c: c.ncc) if has_tmpl else (lambda c: c.range_db)
    eligible = [c for c in cands
                if (not has_tmpl or c.ncc >= NCC_ACCEPT) and c.range_db >= MIN_PROM_DB]

    if eligible:
        if has_tmpl:
            # Among co-confident clicks (NCC within NCC_COMPETE_DELTA of the best), the cut
            # point is the PRESS = the EARLIEST one (protocol: 切点 = down onset). This is
            # robust to the down/up loudness flip seen in the data, and keeps the release
            # (which follows within DOWNUP_MAX_MS) from being mistaken for the onset.
            top = max(c.ncc for c in eligible)
            peers = [c for c in eligible if c.ncc >= top - NCC_COMPETE_DELTA]
            winner = min(peers, key=lambda c: c.onset)
            band = round(DOWNUP_MAX_MS * sr / 1000.0)
            reasons: list[str] = []
            if any(c.onset - winner.onset > band for c in peers):   # a 2nd, separate click
                reasons.append("multi_candidate")
            weak = winner.ncc < QUALITY_NCC or winner.range_db < QUALITY_DB
            if weak:                            # weak match/prominence -> not auto-confident
                reasons.append("low_conf")
            return Pick(winner, "存疑" if weak else "确定", tuple(reasons), cands)

        # no template: prominence-only, loudest wins
        winner = max(eligible, key=lambda c: c.range_db)
        gap = round(MULTI_GAP_MS * sr / 1000.0)
        reasons = ["multi_candidate"] if any(abs(c.onset - winner.onset) > gap
                                             for c in eligible if c is not winner) else []
        return Pick(winner, "确定", tuple(reasons), cands)

    winner = max(cands, key=key)
    reasons = []
    if not has_tmpl:
        reasons.append("no_template")
    elif winner.ncc < NCC_ACCEPT:
        reasons.append("low_ncc")
    if winner.range_db < MIN_PROM_DB:
        reasons.append("low_prom")
    return Pick(winner, "存疑", tuple(reasons), cands)


# --- file-level passes (template self-calibration, then localization) ----------

@dataclass
class WindowSpec:
    click_idx: int                 # 1-based click slot (Q index for a QN clip)
    t_lo: float | None             # seconds; None = whole clip (QN)
    t_hi: float | None
    sheet_flags: tuple[str, ...] = ()


def clamped_bounds(windows: list["WindowSpec"], sr: int, n: int,
                   margin_ms: float = MARGIN_MS) -> list[tuple[int, int]]:
    """Margin-expanded window bounds, each clamped to the midpoints with its neighbouring
    annotated clicks so one click's window never reaches into the adjacent click (the same
    neighbour-clamp idea as snap.windows_from_marks). Whole-clip (QN) windows and windows
    with no real-time neighbours are returned unclamped."""
    centers = [((w.t_lo + w.t_hi) / 2.0) if w.t_lo is not None else None for w in windows]
    real = sorted(c for c in centers if c is not None)
    out: list[tuple[int, int]] = []
    for w, c in zip(windows, centers):
        lo, hi = window_bounds(w.t_lo, w.t_hi, sr, n, margin_ms)
        if c is not None and len(real) > 1:
            i = bisect.bisect_left(real, c)
            prev_c = next((real[j] for j in range(i - 1, -1, -1) if real[j] < c), None)
            nxt_c = next((real[j] for j in range(i, len(real)) if real[j] > c), None)
            if prev_c is not None:
                lo = max(lo, int((prev_c + c) / 2.0 * sr))
            if nxt_c is not None:
                hi = min(hi, int((c + nxt_c) / 2.0 * sr))
            if hi <= lo:                      # overlapping annotations -> small window at center
                cc = int(c * sr)
                lo, hi = max(cc - int(0.1 * sr), 0), min(cc + int(0.1 * sr), n - 1)
        out.append((lo, hi))
    return out


def load_detect(wav: Path) -> tuple[np.ndarray, int, np.ndarray]:
    """Load a WAV and run the pipeline detector once. Returns (env, sr, global_peaks)."""
    y, sr = dsp.load(wav)
    peaks, env, _ = es.detect_candidates(y, sr)
    return env, sr, peaks


def template_segments(env: np.ndarray, sr: int, peaks: np.ndarray,
                      windows: list[WindowSpec]) -> list[np.ndarray]:
    """First pass: per window, snap the loudest candidate; if prominent (>=12 dB),
    return its L2-normalized envelope segment for the global template."""
    n = len(env)
    segs: list[np.ndarray] = []
    for w, (lo, hi) in zip(windows, clamped_bounds(windows, sr, n)):
        cands = window_candidates(env, sr, lo, hi, peaks)
        if not cands:
            continue
        best = max(cands, key=lambda p: env[p])         # loudest transient = first-pass click
        res = snap.snap_v2_one(env, (max(best - _HALF, 0), int(best)), sr)
        if res.range_db >= MIN_PROM_DB:
            u = verify.l2_normalize(verify.extract_segment(env, int(res.sample)))
            if u is not None:
                segs.append(u)
    return segs


def build_global_template(seg_lists: list[list[np.ndarray]]) -> tuple[np.ndarray | None, int]:
    """Per-sample median of all prominent first-pass segments, zero-meaned + unit-norm
    (same construction as verify.build_template, pooled across the whole batch)."""
    segs = [s for lst in seg_lists for s in lst]
    if not segs:
        return None, 0
    med = np.median(np.asarray(segs), axis=0)
    tmpl = verify.l2_normalize(med - med.mean())
    return (tmpl, len(segs)) if tmpl is not None else (None, 0)


@dataclass
class WinResult:
    spec: WindowSpec
    pick: Pick


def locate_file(env: np.ndarray, sr: int, peaks: np.ndarray,
                windows: list[WindowSpec], tmpl: np.ndarray | None) -> list[WinResult]:
    """Second pass: localize every window of one file against the global template."""
    n = len(env)
    results: list[WinResult] = []
    for w, (lo, hi) in zip(windows, clamped_bounds(windows, sr, n)):
        results.append(WinResult(w, pick_click(env, sr, lo, hi, peaks, tmpl)))
    return results


# --- output writers ------------------------------------------------------------

def write_located_csv(stem: str, results: list[WinResult], sr: int) -> Path:
    """labels/<stem>.located.csv — protocol 4-col (sample,time_s,type,confidence);
    one `down` row per located window (winner only)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / f"{stem}.located.csv"
    rows = sorted((r.pick.winner.onset, r.pick.confidence)
                  for r in results if r.pick.winner is not None)
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("sample,time_s,type,confidence\n")
        for s, conf in rows:
            f.write(f"{s},{s / sr:.6f},down,{conf}\n")
    return out


def write_cuts_csv(stem: str, results: list[WinResult], sr: int, n: int,
                   basis: str = "sheet-window") -> tuple[Path, list[int], list[int]]:
    """labels/<stem>.cuts.csv — 5-col, the cut.py input. Returns (path, kept, dropped).

    Cut points must be strictly increasing and strictly inside (0, n) for
    cut.segment_bounds, so we sort + dedup winners and drop any at the file edges
    (dropped samples are reported, never silently lost)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / f"{stem}.cuts.csv"
    wins = [r.pick.winner for r in results if r.pick.winner is not None]
    confs = {w.onset: r.pick.confidence
             for r in results if (w := r.pick.winner) is not None}
    rdb = {w.onset: w.range_db for w in wins}
    kept: list[int] = []
    dropped: list[int] = []
    for s in sorted({w.onset for w in wins}):
        if 0 < s < n and (not kept or s > kept[-1]):
            kept.append(s)
        else:
            dropped.append(s)
    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("sample,time_s,confidence,range_db,basis\n")
        for s in kept:
            f.write(f"{s},{s / sr:.6f},{confs.get(s, '确定')},{rdb.get(s, 0.0):.1f},{basis}\n")
    return out, kept, dropped


def overlay_figure(stem: str, env: np.ndarray, sr: int,
                   results: list[WinResult]) -> Path:
    """figures/<stem>_locate_overlay.png — one panel per window for human sign-off.

    Per panel (log-y envelope around the chosen onset): green/red line = onset
    (red = 存疑), magenta dot = peak, grey dots = other candidates; title carries the
    click slot, NCC, prominence and any flags. Mirrors guardrails.overlay_snaps."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    view = int(0.20 * sr)
    cols = 7
    rows = max(1, int(np.ceil(len(results) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 1.8 * rows), squeeze=False)
    n = len(env)
    n_doubt = 0
    for k, r in enumerate(results):
        ax = axes[k // cols][k % cols]
        p = r.pick
        if p.winner is None:
            ax.set(xticks=[], yticks=[], title=f"c{r.spec.click_idx}: NONE")
            ax.title.set_fontsize(6)
            n_doubt += 1
            continue
        doubt = p.confidence == "存疑"
        n_doubt += int(doubt)
        o = p.winner.onset
        a, b = max(o - view, 0), min(o + view, n - 1)
        t = (np.arange(a, b) - o) / sr * 1000.0
        ax.semilogy(t, np.maximum(env[a:b], 1e-9), lw=0.6, color="0.35")
        for c in p.candidates:                          # other candidates, faint
            if a <= c.onset < b and c is not p.winner:
                ax.plot((c.onset - o) / sr * 1000.0, max(env[c.onset], 1e-9),
                        ".", color="0.7", ms=3)
        ax.axvline(0, color="r" if doubt else "g", lw=1.0)
        ax.plot((p.winner.peak - o) / sr * 1000.0, max(p.winner.peak_amp, 1e-9), "m.", ms=4)
        fl = ("\n" + ",".join(p.reasons)) if p.reasons else ""
        ax.set(xticks=[], yticks=[],
               title=f"c{r.spec.click_idx} ncc{p.winner.ncc:.2f} {p.winner.range_db:.0f}dB{fl}")
        ax.title.set_fontsize(6)
    for k in range(len(results), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{stem}: located clicks (green=onset, red=doubt, m.=peak, grey=other cand) "
                 f"| {n_doubt}/{len(results)} flagged", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"{stem}_locate_overlay.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
