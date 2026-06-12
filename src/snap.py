"""snap_v1 — deterministic snapping of a coarse human mark to the precise onset sample.

Implements the FROZEN rule in annotation-protocol.md §3 (the only authority). The
version string below pins every parameter; changing any of them is snap_v2 and
requires re-annotating everything (see the protocol's freeze constraint).

Pipeline role: workflow §7 step 4. A human places coarse marks (~1 ms scale), then
snap_v1 deterministically pins each to the energy-rise "foot" (NOT the peak), so
~2500 clicks are consistent and free of cross-annotator ±1-2 ms jitter.

snap_one is TYPE-AGNOSTIC: it snaps `down` and `up` rows identically (both are
energy onsets). down/up pairing/assignment is export_seeds' job, not snap's.

Anti-circularity (protocol §3.1 #2): if pipeline step-6 backtrack reuses snap_v1,
the eval "precision" then reflects detection + peak-picking only — keep them
logically independent, or annotate the eval that the number excludes backtrack.

Usage:
    python src/snap.py <seed_or_label.csv> <wav> [-o out.csv] [--thresh-db 6]
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

# --- frozen snap_v1 parameters (annotation-protocol.md §3) ---------------------
FLOOR_LO_MS, FLOOR_HI_MS = -110.0, -10.0   # floor-median window, before coarse mark
MIN_FLOOR_WIN_MS = 30.0                     # usable window < this -> whole-file median
SCAN_LO_MS, SCAN_HI_MS = -10.0, +10.0       # onset scan window around coarse mark
ARM_INIT_LO_MS, ARM_INIT_HI_MS = -11.0, -10.0  # 1 ms window to init arm state
ARM_MS = 1.0                                # continuous-below duration to become "armed"
THRESH_DB = 6.0                             # floor + 6 dB

SNAP_VERSION = (
    "snap_v1: HP2k / env1ms-RMS / floor[-110,-10]ms-median / +6dB / arm1ms / win±10ms"
)


@dataclass(frozen=True)
class SnapResult:
    sample: int        # snapped onset sample
    confidence: str    # "确定" | "存疑"
    floor: float       # linear RMS floor used
    threshold: float   # linear floor * db_gain(thresh_db)
    fallback: bool     # True if the no-crossing local-min fallback was taken


def _off(coarse: int, ms_value: float, sr: int) -> int:
    """Coarse-relative offset in ms -> absolute sample index (unclamped)."""
    return coarse + round(ms_value * sr / 1000.0)


def snap_one(env: np.ndarray, coarse: int, sr: int = SR,
             thresh_db: float = THRESH_DB) -> SnapResult:
    """Snap one coarse mark against a PRE-COMPUTED canonical envelope.

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
             thresh_db: float = THRESH_DB) -> list[SnapResult]:
    """Snap every mark in a seed/label CSV; write a protocol-format CSV.

    Envelope is computed ONCE for the file. Output is re-sorted ascending by sample.
    A fallback row is forced to 存疑; otherwise the incoming confidence is preserved
    (so §5.1 doubt-band 存疑 survives snapping).
    """
    y, sr = dsp.load(wav)
    env = dsp.hp_rms_envelope(y, sr)
    rows = _read_rows(in_csv)

    out = []
    for r in rows:
        res = snap_one(env, int(r["sample"]), sr, thresh_db)
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
          f"{n_fb} fallback (存疑)  [thresh +{thresh_db:g} dB]")
    return [res for res, _, _ in out]


def main() -> None:
    ap = argparse.ArgumentParser(description="snap_v1: snap coarse marks to onset samples.")
    ap.add_argument("in_csv", type=Path, help="seed/label CSV (sample,time_s,type,confidence)")
    ap.add_argument("wav", type=Path, help="matching data/wav/<name>.wav")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV (default: <in_csv stem>.snapped.csv next to input)")
    ap.add_argument("--thresh-db", type=float, default=THRESH_DB,
                    help="threshold above floor in dB (default 6; 10 only for sensitivity check)")
    args = ap.parse_args()
    out = args.out or args.in_csv.with_suffix(".snapped.csv")
    snap_csv(args.in_csv, args.wav, out, args.thresh_db)


if __name__ == "__main__":
    main()
