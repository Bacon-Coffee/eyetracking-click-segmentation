"""Export per-file SEED labels for human review in Sonic Visualiser.

Pipeline role: workflow §7 step 1-2 (annotation-protocol.md). Produces COARSE,
pre-snap candidates so a human can confirm/add/delete/coarse-adjust, after which
`snap.py` (step 4) refines each mark to the exact onset sample.

Stages:
  1. Coarse peak detect            -> reuse interference_analysis.detect_peaks
  2. Reject 7-10 ms periodic trains-> reuse interference_analysis.group_trains
  3. Min-gap merge (30 ms)         -> peaks closer than the protocol's safe
                                      click-to-click floor collapse to the strongest
                                      (kills duplicate/split candidates + residual
                                      6-11 ms interference pairs that escaped #2)
  4. §5.1 deterministic down/up    -> greedy pairing on the CANONICAL HP2k/1ms-RMS
     assignment                       envelope amplitude (dsp.hp_rms_envelope)
  5. Write labels/<name>.seed.csv  -> protocol-format CSV, deduped, one row per
                                      candidate transient, positioned at the
                                      snap_v2-style onset so SV review marks sit
                                      right on the foot (final positions are still
                                      recomputed by snap.py AFTER human confirmation)

The §5.1 amplitude MUST be the same envelope snap_v1 uses — see dsp.hp_rms_envelope.
The mean-abs envelope is used ONLY to drive detect_peaks (its 0.12*max threshold is
calibrated to that scale); all down/up amplitude decisions read the RMS envelope.

Sonic Visualiser import: open the seed CSV as an annotation layer with
"Timing is specified: in audio sample frames", map column 1 (sample) -> time,
column 3 (type) -> label, sample rate 48000. Edit, then export; feed the export to
`from_sv_export` -> `snap.snap_csv` to produce final labels/<name>.csv.

Usage:
    python src/export_seeds.py                 # all data/wav/*.wav -> labels/*.seed.csv
    python src/export_seeds.py BJ_C3           # one file by stem
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import dsp
import interference_analysis as ia
import snap

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "data" / "wav"
LABELS_DIR = REPO / "labels"

# §5 / §5.1 constants
GAP_LO_MS, GAP_HI_MS = 60.0, 150.0    # down->up pairing window
UP_MARGIN_DB = 6.0                    # >= this weaker than pending down -> up (确定)
DOUBT_LO_DB = 3.0                     # [DOUBT_LO_DB, UP_MARGIN_DB) -> down (存疑)
_EPS = 1e-12

_FIELDS = ["sample", "time_s", "type", "confidence"]

# Protocol §5: click-to-click min_gap ~30-50 ms is safe. Peaks closer than this are
# duplicates / split transients / residual interference -> merge, keep the strongest.
# (Safe for real down->up pairs too: those sit at 60-150 ms.)
MIN_PEAK_GAP_MS = 30.0


def merge_close_peaks(peaks: np.ndarray, env_rms: np.ndarray, sr: int,
                      min_gap_ms: float = MIN_PEAK_GAP_MS) -> np.ndarray:
    """Collapse runs of peaks closer than min_gap to the single strongest peak.

    Fixes the v1-era seed pollution found in review: duplicate rows (multiple peaks
    backtracked onto one sample) and residual 6-11 ms interference pairs. Greedy
    left-to-right: a peak within min_gap of the current group joins it; each group
    emits its argmax (by canonical RMS amplitude).
    """
    if len(peaks) == 0:
        return peaks
    gap = round(min_gap_ms * sr / 1000.0)
    peaks = np.sort(np.asarray(peaks, dtype=int))
    out, group = [], [peaks[0]]
    for p in peaks[1:]:
        if p - group[-1] < gap:
            group.append(p)
        else:
            out.append(max(group, key=lambda q: env_rms[q]))
            group = [p]
    out.append(max(group, key=lambda q: env_rms[q]))
    return np.asarray(out, dtype=int)


def detect_candidates(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coarse click candidates (PEAKS): detect -> drop trains -> min-gap merge.

    Returns (peaks, env_rms, stats). HP is computed once and shared: the mean-abs
    envelope drives detect_peaks (calibrated to its 0.12*max threshold); the RMS
    envelope is the canonical §5.1 amplitude AND the merge/seed-positioning surface.
    """
    hp = dsp.highpass(y, sr)                 # identical params to ia.highpass
    env_d = ia.envelope(hp, sr)              # mean-abs, for detect_peaks only
    env_rms = dsp.rms_envelope(hp, sr)       # canonical §5.1 / snap_v2 amplitude
    peaks = ia.detect_peaks(env_d, sr)
    mask = ia.group_trains(peaks, sr)        # True = periodic-train member
    clicks = peaks[~mask]
    merged = merge_close_peaks(clicks, env_rms, sr)
    stats = {"n_peaks": int(len(peaks)), "n_train": int(mask.sum()),
             "n_clicks": int(len(merged)), "n_merged_away": int(len(clicks) - len(merged))}
    return merged, env_rms, stats


def coarse_onset(env: np.ndarray, peak: int, sr: int) -> int:
    """Seed position for a detected peak = snap_v2-style onset (peak -20 dB foot).

    Positions the SV review mark right on the foot of the transient, so the human
    mostly just confirms. NOT circular in the eval sense: final label positions are
    recomputed by snap.py from the HUMAN-CONFIRMED marks; this only chooses where
    the to-be-confirmed mark is displayed. (detect_peaks peaks come from the
    mean-abs envelope, so re-argmax on the RMS envelope inside snap_v2_one also
    cleans up the peak location.)
    """
    lo = max(int(peak) - round(snap.WIN_HALF_MS * sr / 1000.0), 0)
    return snap.snap_v2_one(env, (lo, int(peak)), sr).sample


def assign_down_up(peaks: np.ndarray, env_rms: np.ndarray, sr: int,
                   gap_lo_ms: float = GAP_LO_MS, gap_hi_ms: float = GAP_HI_MS,
                   up_margin_db: float = UP_MARGIN_DB,
                   doubt_lo_db: float = DOUBT_LO_DB) -> list[tuple[int, str, str]]:
    """§5.1 greedy left-to-right down/up assignment. Returns [(onset,type,confidence)].

    Per §5.1 the input transient = (onset sample, peak envelope amplitude): timing /
    gaps use the coarse ONSET, the dB comparison uses the PEAK amplitude (click
    strength). The emitted sample is the seed onset (snap_v2 recomputes it from the
    human-confirmed marks later).

    For transient T with a pending down D and (onset_T - onset_D) in [60,150] ms, let
    ΔdB = 20*log10(peakamp(D)/peakamp(T))  (positive => T weaker). Three-tier rule:
      ΔdB >= 6        -> up   (确定), pair & close D
      3 <= ΔdB < 6    -> down (存疑), doubt band — default down so a real click is
                          never hidden inside an up; flag for human review
      ΔdB < 3         -> down (确定), open a new pending down
    No pending D, or gap out of [60,150] ms -> down (确定).
    """
    onsets = [coarse_onset(env_rms, int(p), sr) for p in peaks]
    amps = [float(env_rms[int(p)]) for p in peaks]

    out: list[tuple[int, str, str]] = []
    pending: int | None = None      # index into peaks/onsets of the open down
    for k in range(len(peaks)):
        on = onsets[k]
        assigned = False
        if pending is not None:
            gap_ms = (on - onsets[pending]) / sr * 1000.0
            if gap_lo_ms <= gap_ms <= gap_hi_ms:
                d_db = dsp.amp_to_db_ratio(amps[pending] + _EPS, amps[k] + _EPS)
                if d_db >= up_margin_db:
                    out.append((on, "up", "确定"))
                    pending = None
                elif d_db >= doubt_lo_db:
                    out.append((on, "down", "存疑"))  # doubt band -> down, flagged
                    pending = k
                else:
                    out.append((on, "down", "确定"))
                    pending = k
                assigned = True
        if not assigned:
            out.append((on, "down", "确定"))
            pending = k
    return out


def write_seed_csv(rows: list[tuple[int, str, str]], out_csv: Path, sr: int) -> None:
    """Write protocol-format CSV, sorted, EXACT-DUPLICATE samples dropped (keep first)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r[0])
    seen: set[int] = set()
    rows = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        for sample, typ, conf in rows:
            w.writerow([sample, f"{sample / sr:.6f}", typ, conf])


def export_one(wav: Path) -> dict:
    """Detect -> reject trains -> §5.1 assign -> write labels/<stem>.seed.csv."""
    y, sr = dsp.load(wav)
    clicks, env_rms, stats = detect_candidates(y, sr)
    rows = assign_down_up(clicks, env_rms, sr)
    out_csv = LABELS_DIR / f"{wav.stem}.seed.csv"
    write_seed_csv(rows, out_csv, sr)

    n_down = sum(1 for _, t, _ in rows if t == "down")
    n_up = sum(1 for _, t, _ in rows if t == "up")
    n_doubt = sum(1 for _, _, c in rows if c == "存疑")
    stats.update({"n_down": n_down, "n_up": n_up, "n_doubt": n_doubt, "out": out_csv})
    return stats


def from_sv_export(sv_csv: Path, out_csv: Path, sr: int = dsp.SR,
                   col_is_frames: bool = True) -> int:
    """Rebuild a protocol CSV from a Sonic Visualiser annotation export.

    SV exports a time-instants layer as `<time>,<value>,<label>` (no header). With
    the documented frames-mode import, column 0 is the sample frame. Human-added
    marks default to confidence=确定 (snap.py's fallback re-flags 存疑 as needed).
    Pass col_is_frames=False if the layer was exported in seconds.
    """
    rows: list[tuple[int, str, str]] = []
    with open(sv_csv, newline="", encoding="utf-8") as f:
        for rec in csv.reader(f):
            if not rec:
                continue
            c0 = rec[0].strip()
            try:
                v = float(c0)
            except ValueError:
                continue  # skip a stray header line
            sample = int(round(v if col_is_frames else v * sr))
            label = rec[-1].strip().strip('"') if len(rec) >= 2 else "down"
            typ = label if label in ("down", "up") else "down"
            rows.append((sample, typ, "确定"))
    write_seed_csv(rows, out_csv, sr)
    return len(rows)


def main(argv: list[str]) -> None:
    if argv:
        wavs = [WAV_DIR / f"{a}.wav" if not a.endswith(".wav") else Path(a) for a in argv]
    else:
        wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        print(f"No WAVs found in {WAV_DIR}", file=sys.stderr)
        return

    hdr = (f"{'file':10s}{'peaks':>7s}{'train':>7s}{'merged':>8s}{'cand':>7s}"
           f"{'down':>7s}{'up':>6s}{'存疑':>6s}")
    print(hdr)
    print("-" * (len(hdr) + 2))
    for wav in wavs:
        s = export_one(wav)
        print(f"{wav.stem:10s}{s['n_peaks']:7d}{s['n_train']:7d}{s['n_merged_away']:8d}"
              f"{s['n_clicks']:7d}{s['n_down']:7d}{s['n_up']:6d}{s['n_doubt']:6d}")
    print("\ndown ~= candidate click count (after train removal + 30 ms min-gap merge);")
    print("still an over-detection upper bound — the human prunes during review.")
    print("Seeds written to labels/<name>.seed.csv — review in Sonic Visualiser, then snap.")


if __name__ == "__main__":
    main(sys.argv[1:])
