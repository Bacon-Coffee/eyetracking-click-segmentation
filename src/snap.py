"""snap_v2 — deterministic snapping of a confirmed human mark to the precise onset sample.

Implements annotation-protocol.md §3 (the only authority). Core idea: the human
guarantees "one mark <-> exactly one click" (count prior, +-20 ms placement); the
rule takes the STRONGEST envelope peak in a neighbor-clamped window around the
mark, then backtracks from that peak to the onset foot at peak - X dB. No absolute
floor, no arm state: dense passages and the real ~15-25 ms pre-onset sub-transient
(precursor) cannot break it — the precursor is weaker than the main peak, so
argmax skips it, and the backtrack stops at the main rise as soon as the dip
between precursor and main rise drops below peak - X dB.

snap_v2 is TYPE-AGNOSTIC: `down` and `up` rows snap identically (both are energy
onsets). down/up pairing/assignment is export_seeds' job, not snap's.

snap_v1 (floor + 6 dB / arm 1 ms / +-10 ms) FAILED the §3.1 guardrail (60-62%
fallback: no quiet foot to arm against, contaminated floor in dense passages) and
was retired before any real annotation. Its code is kept below ONLY so guardrails
can compare rules; do not use it for labels.

Anti-circularity (protocol §3.1 #2): pipeline step 5/6 backtrack+cut must not
silently reuse this same peak-relative logic — keep them independent, or annotate
the eval that the precision number excludes backtrack.

Usage:
    python src/snap.py <seed_or_label.csv> <wav> [-o out.csv] [--backtrack-db 20]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

import dsp
from dsp import SR

# --- snap_v2 parameters (annotation-protocol.md §3) ----------------------------
WIN_HALF_MS = 30.0        # search half-window around the mark, clamped to neighbor midpoints
BACKTRACK_MAX_MS = 15.0   # how far left of the peak the backtrack may walk
MID_FRAC = 0.5            # backtrack depth = MID_FRAC * pre-peak dynamic range (dB) ...
CAP_DB = 20.0             # ... capped at CAP_DB below the peak
MIN_PROMINENCE_DB = 12.0  # pre-peak range below this -> 存疑 (weak/glued transient)

# Why adaptive: calibration on this data showed the pre-peak 15 ms dynamic range is
# only ~16 dB at the median (p25 ~11.5 dB) — a FIXED 20 dB depth has no crossing for
# ~60% of clicks, and any fixed X either starves (deep) or rides noise (shallow).
# The dB-midpoint crossing always exists, so there is NO fallback branch; doubt is
# instead flagged by low prominence, which is what actually predicts a bad foot.

SNAP_VERSION = (
    "snap_v2: HP2k / env1ms-RMS / win±30ms-clamp / peak-argmax / "
    "bt15ms-mid0.5-cap20dB / prom>=12dB"
)

# --- snap_v1 parameters — RETIRED (failed §3.1; kept only for guardrail comparison)
FLOOR_LO_MS, FLOOR_HI_MS = -110.0, -10.0   # floor-median window, before coarse mark
MIN_FLOOR_WIN_MS = 30.0                     # usable window < this -> whole-file median
SCAN_LO_MS, SCAN_HI_MS = -10.0, +10.0       # onset scan window around coarse mark
ARM_INIT_LO_MS, ARM_INIT_HI_MS = -11.0, -10.0  # 1 ms window to init arm state
ARM_MS = 1.0                                # continuous-below duration to become "armed"
THRESH_DB = 6.0                             # floor + 6 dB

SNAP_V1_VERSION = (
    "snap_v1 [RETIRED]: HP2k / env1ms-RMS / floor[-110,-10]ms-median / +6dB / arm1ms / win±10ms"
)


@dataclass(frozen=True)
class SnapResult:
    sample: int        # snapped onset sample
    confidence: str    # "确定" | "存疑"
    floor: float       # linear RMS floor used
    threshold: float   # linear floor * db_gain(thresh_db)
    fallback: bool     # True if the no-crossing local-min fallback was taken


@dataclass(frozen=True)
class SnapV2Result:
    sample: int        # snapped onset sample
    confidence: str    # "确定" | "存疑"
    peak: int          # main-transient peak sample (window argmax)
    peak_amp: float    # linear RMS envelope amplitude at peak
    threshold: float   # the crossing level actually used (linear)
    range_db: float    # pre-peak dynamic range: peak vs 15 ms-window minimum
    win: tuple[int, int]  # the neighbor-clamped search window actually used
    fallback: bool     # True = low prominence (range_db < MIN_PROMINENCE_DB) -> 存疑


def _off(coarse: int, ms_value: float, sr: int) -> int:
    """Coarse-relative offset in ms -> absolute sample index (unclamped)."""
    return coarse + round(ms_value * sr / 1000.0)


# --- snap_v2 (operative rule) ---------------------------------------------------

def windows_from_marks(marks: list[int], n: int, sr: int = SR,
                       half_ms: float = WIN_HALF_MS) -> list[tuple[int, int]]:
    """Neighbor-clamped, non-overlapping search windows, one per SORTED mark.

    W_k = [max(m_k - half, mid(m_{k-1}, m_k)), min(m_k + half, mid(m_k, m_{k+1}))]
    clamped to [0, n-1]. Clamping needs ALL marks of the file (down AND up), so the
    window of a mark never reaches into a neighbour's transient — this is what turns
    the human's "exactly one click per mark" count prior into a per-window guarantee.
    """
    if any(marks[i] > marks[i + 1] for i in range(len(marks) - 1)):
        raise ValueError("marks must be sorted ascending")
    half = round(half_ms * sr / 1000.0)
    out = []
    for k, m in enumerate(marks):
        lo = max(m - half, 0)
        hi = min(m + half, n - 1)
        if k > 0:
            lo = max(lo, (marks[k - 1] + m) // 2 + 1)   # +1: previous window owns the midpoint
        if k + 1 < len(marks):
            hi = min(hi, (m + marks[k + 1]) // 2)
        if hi < lo:                        # degenerate (duplicate marks) — stay put
            lo = hi = min(max(m, 0), n - 1)
        out.append((lo, hi))
    return out


def snap_v2_one(env: np.ndarray, win: tuple[int, int], sr: int = SR,
                mid_frac: float = MID_FRAC, cap_db: float = CAP_DB) -> SnapV2Result:
    """Snap one mark: window argmax = main-transient peak, then backtrack to the foot.

    ``env`` MUST be ``dsp.hp_rms_envelope(y)``. Over B = [peak - 15 ms, peak]:
    m = argmin(env[B]); R = dB range peak/m; depth D = min(mid_frac * R, cap_db);
    onset = nearest up-crossing of peak - D dB left of the peak (guaranteed to exist
    since env[m] < thr <= peak). The weaker precursor is skipped because the dip
    between it and the main rise sits below the dB-midpoint. confidence = 存疑 iff
    R < MIN_PROMINENCE_DB. Default runs MUST use mid_frac 0.5 (the version string is
    fixed); other values are only for the §3.1 sensitivity guardrail.
    """
    lo, hi = win
    peak = lo + int(np.argmax(env[lo:hi + 1]))
    peak_amp = float(env[peak])

    bt_lo = max(peak - round(BACKTRACK_MAX_MS * sr / 1000.0), 0)
    m_amp = float(np.min(env[bt_lo:peak + 1]))
    range_db = dsp.amp_to_db_ratio(peak_amp, max(m_amp, 1e-12))
    depth_db = min(mid_frac * range_db, cap_db)
    thr = peak_amp * dsp.db_gain(-depth_db)       # linear domain (see dsp.db_gain)
    doubt = range_db < MIN_PROMINENCE_DB

    onset = bt_lo                                  # degenerate flat case (range ~ 0)
    for j in range(peak - 1, bt_lo - 1, -1):      # walk left, nearest crossing first
        if env[j] < thr:
            onset = j + 1
            break
    return SnapV2Result(onset, "存疑" if doubt else "确定", peak, peak_amp, thr,
                        float(range_db), (lo, hi), doubt)


def snap_v2_marks(env: np.ndarray, marks: list[int], sr: int = SR,
                  mid_frac: float = MID_FRAC, cap_db: float = CAP_DB) -> list[SnapV2Result]:
    """Batch snap_v2 over ALL marks of one file (windows need every mark, see above)."""
    wins = windows_from_marks(marks, len(env), sr)
    return [snap_v2_one(env, w, sr, mid_frac, cap_db) for w in wins]


def snap_one(env: np.ndarray, coarse: int, sr: int = SR,
             thresh_db: float = THRESH_DB) -> SnapResult:
    """[RETIRED snap_v1 — guardrail comparison only] Snap one coarse mark.

    ``env`` MUST be ``dsp.hp_rms_envelope(y)`` — passing the envelope (not raw audio)
    lets a batch driver compute HP+RMS once per file and snap hundreds of marks, and
    lets the §3.1 sensitivity check vary ``thresh_db`` without recomputing. Default
    runs MUST use 6 dB (the version string is fixed); ``thresh_db=10`` is only for
    the sensitivity guardrail.
    """
    n = len(env)
    arm_samples = max(1, round(ARM_MS * sr / 1000.0))  # 48 @ 48k

    # --- Step A: floor (linear RMS amplitude), median over [-110,-10] ms ---------
    flo = max(_off(coarse, FLOOR_LO_MS, sr), 0)
    fhi = max(_off(coarse, FLOOR_HI_MS, sr), 0)
    min_win = round(MIN_FLOOR_WIN_MS * sr / 1000.0)
    if fhi - flo < min_win:                 # near file start -> whole-file median
        floor = float(np.median(env))
    else:
        floor = float(np.median(env[flo:fhi]))

    # --- Step B: threshold (linear; never compute in dB, floor may be ~0) --------
    threshold = floor * dsp.db_gain(thresh_db)

    # --- Step C: scan window [-10,+10] ms, clamped ------------------------------
    slo = max(_off(coarse, SCAN_LO_MS, sr), 0)
    shi = min(_off(coarse, SCAN_HI_MS, sr), n - 1)

    # --- Step D: arm-state machine ----------------------------------------------
    a0 = max(_off(coarse, ARM_INIT_LO_MS, sr), 0)
    a1 = max(_off(coarse, ARM_INIT_HI_MS, sr), 0)
    armed = a1 > a0 and bool(np.all(env[a0:a1] < threshold))
    below_count = (a1 - a0) if armed else 0

    for i in range(slo, shi + 1):
        if env[i] < threshold:
            below_count += 1
            if below_count >= arm_samples:
                armed = True
        else:  # upward sample (env >= threshold)
            if armed:
                return SnapResult(i, "确定", floor, threshold, fallback=False)
            below_count = 0  # crossing before arming doesn't count

    # --- Step E: fallback — nearest preceding envelope local min, mark 存疑 -------
    seg = env[slo:coarse + 1]
    j = None
    if len(seg) >= 3:
        mins = argrelmin(seg)[0]
        if len(mins):
            j = slo + int(mins[-1])         # nearest-preceding local min
    if j is None:
        j = slo + int(np.argmin(seg)) if len(seg) else coarse
    return SnapResult(j, "存疑", floor, threshold, fallback=True)


def snap_one_wav(y: np.ndarray, coarse: int, sr: int = SR,
                 thresh_db: float = THRESH_DB) -> SnapResult:
    """Single-shot convenience: compute the canonical envelope then snap. For tests."""
    return snap_one(dsp.hp_rms_envelope(y, sr), coarse, sr, thresh_db)


# --- batch / CLI ---------------------------------------------------------------

_FIELDS = ["sample", "time_s", "type", "confidence"]


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def snap_csv(in_csv: Path, wav: Path, out_csv: Path,
             mid_frac: float = MID_FRAC) -> list[SnapV2Result]:
    """snap_v2 every mark in a confirmed seed/label CSV; write a protocol-format CSV.

    Envelope is computed ONCE for the file; windows are clamped against ALL marks
    (down + up) of the file. A fallback row is forced to 存疑; otherwise the incoming
    confidence is preserved (so §5.1 doubt-band 存疑 survives snapping).
    """
    y, sr = dsp.load(wav)
    env = dsp.hp_rms_envelope(y, sr)
    rows = sorted(_read_rows(in_csv), key=lambda r: int(r["sample"]))

    marks = [int(r["sample"]) for r in rows]
    results = snap_v2_marks(env, marks, sr, mid_frac)

    out = []
    for r, res in zip(rows, results):
        conf = "存疑" if res.fallback else r.get("confidence", "确定")
        out.append((res, r.get("type", "down"), conf))

    out.sort(key=lambda t: t[0].sample)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for res, typ, conf in out:
            w.writerow([res.sample, f"{res.sample / sr:.6f}", typ, conf])

    n_fb = sum(1 for res, _, _ in out if res.fallback)
    print(f"{SNAP_VERSION}")
    print(f"snapped {len(out)} mark(s) from {in_csv.name} -> {out_csv.name}; "
          f"{n_fb} low-prominence (存疑)  [mid_frac {mid_frac:g}]")
    return [res for res, _, _ in out]


def main() -> None:
    ap = argparse.ArgumentParser(description="snap_v2: snap confirmed marks to onset samples.")
    ap.add_argument("in_csv", type=Path, help="seed/label CSV (sample,time_s,type,confidence)")
    ap.add_argument("wav", type=Path, help="matching data/wav/<name>.wav")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV (default: <in_csv stem>.snapped.csv next to input)")
    ap.add_argument("--mid-frac", type=float, default=MID_FRAC,
                    help="backtrack depth as fraction of pre-peak dB range, capped "
                         "at 20 dB (default 0.5; 0.35/0.65 only for §3.1 sensitivity)")
    args = ap.parse_args()
    out = args.out or args.in_csv.with_suffix(".snapped.csv")
    snap_csv(args.in_csv, args.wav, out, args.mid_frac)


if __name__ == "__main__":
    main()
