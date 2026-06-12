"""Quantify the ~10 ms periodic spike-train interference (confirmed environmental).

For each decoded WAV:
  1. Coarse envelope peak detection (same spirit as notebook: HP 2 kHz + envelope).
  2. Group peaks into "trains": runs of >=5 peaks whose inter-peak intervals are
     5-15 ms with low jitter -> interference; everything else -> click candidates.
  3. Measure the train period precisely (median interval + envelope autocorrelation
     on the densest interference second).
  4. Average spectrum of interference windows vs. non-train (click) windows.
  5. Coverage: fraction of recording time inside interference trains.

Outputs: printed summary + figures/interference_analysis.png
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "data" / "wav"
FIG_DIR = REPO / "figures"
SR = 48000

def load(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr

def highpass(y, sr, fc=2000.0, order=4):
    sos = sps.butter(order, fc / (sr / 2), btype="high", output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)

def envelope(y, sr, smooth_ms=1.0):
    e = np.abs(y)
    n = max(1, int(sr * smooth_ms / 1000))
    return sps.convolve(e, np.ones(n) / n, mode="same")

def detect_peaks(env, sr):
    # Same prominence floor as notebook's detect_clicks (rel_thr=0.12 of env max),
    # but 4 ms min distance instead of 50 ms so periodic-train members are kept.
    thr = 0.12 * float(env.max())
    min_dist = int(sr * 0.004)
    peaks, _ = sps.find_peaks(env, height=thr, distance=min_dist)
    return peaks

def group_trains(peaks, sr, lo_ms=5.0, hi_ms=15.0, min_run=5, jitter=0.25):
    """Return boolean mask over peaks: True = belongs to a periodic train."""
    is_train = np.zeros(len(peaks), dtype=bool)
    if len(peaks) < min_run:
        return is_train
    iv = np.diff(peaks) / sr * 1000.0  # ms
    in_band = (iv >= lo_ms) & (iv <= hi_ms)
    i = 0
    while i < len(iv):
        if not in_band[i]:
            i += 1
            continue
        j = i
        while j < len(iv) and in_band[j]:
            j += 1
        run = iv[i:j]
        # run of consecutive in-band intervals; require length and regularity
        if len(run) + 1 >= min_run and np.std(run) / np.mean(run) < jitter:
            is_train[i : j + 1] = True
        i = j
    return is_train

def avg_spectrum(y, centers, sr, half_ms=25.0, nfft=4096):
    half = int(sr * half_ms / 1000)
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    win = np.hanning(2 * half)
    for c in centers:
        a, b = c - half, c + half
        if a < 0 or b > len(y):
            continue
        seg = y[a:b] * win
        acc += np.abs(np.fft.rfft(seg, nfft))
        cnt += 1
    f = np.fft.rfftfreq(nfft, 1 / sr)
    return f, acc / max(cnt, 1), cnt

def coverage_seconds(peaks, mask, sr, pad_ms=10.0):
    """Total seconds covered by train peaks (union of +-pad windows)."""
    pts = peaks[mask]
    if len(pts) == 0:
        return 0.0, []
    pad = int(sr * pad_ms / 1000)
    ivals = []
    s, e = pts[0] - pad, pts[0] + pad
    for p in pts[1:]:
        if p - pad <= e:
            e = p + pad
        else:
            ivals.append((s, e))
            s, e = p - pad, p + pad
    ivals.append((s, e))
    total = sum(b - a for a, b in ivals) / sr
    return total, ivals

results = {}
fig, axes = plt.subplots(2, 4, figsize=(20, 7))
for col, w in enumerate(sorted(WAV_DIR.glob("*.wav"))):
    y, sr = load(w)
    hp = highpass(y, sr)
    env = envelope(hp, sr)
    peaks = detect_peaks(env, sr)
    mask = group_trains(peaks, sr)
    n_train, n_click = int(mask.sum()), int((~mask).sum())

    iv = np.diff(peaks) / sr * 1000.0
    train_iv = iv[mask[:-1] & mask[1:]] if len(iv) else np.array([])
    period_ms = float(np.median(train_iv)) if len(train_iv) else float("nan")

    cov_s, ivals = coverage_seconds(peaks, mask, sr)
    dur = len(y) / sr

    # autocorrelation on densest interference second for an independent period check
    ac_period = float("nan")
    if len(ivals):
        widths = [(b - a) for a, b in ivals]
        a0, b0 = ivals[int(np.argmax(widths))]
        mid = (a0 + b0) // 2
        seg = env[max(0, mid - sr // 2) : mid + sr // 2].astype(np.float64)
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1 :]
        lo, hi = int(sr * 0.005), int(sr * 0.015)
        if hi < len(ac):
            ac_period = (lo + int(np.argmax(ac[lo:hi]))) / sr * 1000.0

    f_i, S_i, n_i = avg_spectrum(hp, peaks[mask], sr)
    f_c, S_c, n_c = avg_spectrum(hp, peaks[~mask], sr)

    results[w.stem] = dict(dur=dur, peaks=len(peaks), train=n_train, click=n_click,
                           period=period_ms, ac_period=ac_period,
                           cov_s=cov_s, cov_pct=100 * cov_s / dur, n_seg=len(ivals))

    ax = axes[0, col]
    t = peaks / sr
    ax.scatter(t[~mask], np.ones(n_click), s=2, label=f"click cand. ({n_click})")
    ax.scatter(t[mask], np.zeros(n_train), s=2, color="r", label=f"train ({n_train})")
    ax.set(title=f"{w.stem}", xlabel="time (s)", yticks=[])
    ax.legend(fontsize=7, loc="center right")

    ax = axes[1, col]
    if n_i:
        ax.semilogy(f_i, S_i + 1e-9, "r", lw=0.8, label=f"interference avg (n={n_i})")
    ax.semilogy(f_c, S_c + 1e-9, "b", lw=0.8, alpha=0.7, label=f"non-train avg (n={n_c})")
    ax.set(xlim=(0, 16000), xlabel="Hz")
    ax.legend(fontsize=7)

fig.suptitle("Periodic interference vs click candidates (HP 2 kHz)")
fig.tight_layout()
fig.savefig(FIG_DIR / "interference_analysis.png", dpi=90)

hdr = f"{'file':10s}{'dur_s':>7s}{'peaks':>7s}{'train':>7s}{'click':>7s}{'T_ms':>7s}{'T_ac':>7s}{'cov_s':>8s}{'cov_%':>7s}{'n_seg':>7s}"
print(hdr)
print("-" * len(hdr))
for k, r in results.items():
    print(f"{k:10s}{r['dur']:7.1f}{r['peaks']:7d}{r['train']:7d}{r['click']:7d}"
          f"{r['period']:7.2f}{r['ac_period']:7.2f}{r['cov_s']:8.1f}{r['cov_pct']:7.1f}{r['n_seg']:7d}")
