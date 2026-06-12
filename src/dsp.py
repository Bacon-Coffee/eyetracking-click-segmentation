"""Canonical DSP for sample-accurate mouse-click SED — the single source of truth.

Both `snap.py` (snap_v1) and `export_seeds.py` (§5.1 down/up amplitude) import the
high-pass + 1 ms RMS envelope from here, so there is exactly ONE definition of the
"snap_v1 / §5.1 amplitude" envelope (annotation-protocol.md §3 / §5.1).

NOTE on the divergence with `interference_analysis.py`: that module's `envelope()`
is a *mean-abs* moving average, tuned to its own `0.12 * env.max()` peak threshold.
It is intentionally left unchanged. Do NOT reuse it for snapping or amplitude
comparison — use `rms_envelope` / `hp_rms_envelope` below (true RMS).

Canonical envelope = HP 2 kHz (Butterworth-4, zero-phase) -> 1 ms (48-sample)
CENTERED RMS. Centering matches `interference_analysis.envelope`'s convention and
corresponds to the forward-smear / "systematically-slightly-early" bias already
declared in protocol §3 — it is part of snap_v1, not a defect. Switching to a
causal window would change the frozen rule (snap_v2).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps
import soundfile as sf
from scipy.ndimage import uniform_filter1d

# Canonical sample rate — matches decode.py output (48 kHz / mono / float32).
SR = 48_000


def load(path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono. Same contract as interference_analysis.load."""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def highpass(y: np.ndarray, sr: int = SR, fc: float = 2000.0, order: int = 4) -> np.ndarray:
    """Butterworth order-4 high-pass, zero-phase (sosfiltfilt). Canonical HP for snap_v1.

    Zero-phase => no group delay, but symmetric forward smearing (the declared §3 bias).
    """
    sos = sps.butter(order, fc / (sr / 2), btype="high", output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def rms_envelope(y: np.ndarray, sr: int = SR, win_ms: float = 1.0) -> np.ndarray:
    """1 ms (48-sample @48k) CENTERED RMS envelope of ``y``.

    RMS, NOT mean-abs — distinct from interference_analysis.envelope.
    `uniform_filter1d(..., mode="nearest")` centers the window and replicates edges,
    so the envelope is defined right up to sample 0 (clicks near file start).
    """
    n = max(1, round(sr * win_ms / 1000.0))  # 48 @ 48k / 1 ms
    ms = uniform_filter1d(np.square(y, dtype=np.float64), size=n, mode="nearest")
    return np.sqrt(ms).astype(np.float32)


def hp_rms_envelope(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """THE canonical snap_v1 / §5.1 envelope: HP 2 kHz -> 1 ms RMS.

    Everything downstream (snap thresholds, §5.1 peak amplitude) reads this.
    """
    return rms_envelope(highpass(y, sr), sr)


def db_gain(db: float) -> float:
    """Amplitude (not power) dB -> linear factor: 10**(db/20). db_gain(6) ~= 1.9953.

    The envelope is an RMS *amplitude*, so threshold = floor * db_gain(6 dB).
    Do threshold/margin arithmetic in the linear domain (floor may be ~0); use dB
    only for reporting.
    """
    return 10.0 ** (db / 20.0)


def amp_to_db_ratio(a: float, b: float) -> float:
    """20*log10(a/b) — amplitude ratio in dB. For reporting only (guard b>0 upstream)."""
    return 20.0 * np.log10(a / b)
