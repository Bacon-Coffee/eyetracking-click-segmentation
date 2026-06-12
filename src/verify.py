"""verify — pipeline step 4: per-file self-calibrated template matching (NCC).

Scores every coarse `down` candidate against a template the file builds FROM ITSELF
(no human labels): align each candidate at its snap_v2 onset, cut a fixed window from
the HP2k energy envelope, L2-normalize, and take the per-sample MEDIAN over the
"decision-level" subset (isolated >=120 ms AND prominent range_db >=12 dB — the 946
clean candidates from clean_audit). Each down is then scored by ZERO-MEAN normalized
cross-correlation (Pearson r, ±2 ms lag) against that template, the per-file score
distribution is measured, and thresholds are derived from it by a 2-component
Gaussian-mixture posterior (NO hardcoded cutoff; degrades gracefully + flags when a
file isn't bimodal — these seeds are click-DOMINATED, so most files are not cleanly
bimodal and fall to a flagged conservative default). A three-tier decision follows
the existing 存疑-band philosophy, plus sub-30 ms close-pair dedup.

Design authority: docs/annotation-protocol.md (onset = down, §2/§3) and the
clean_audit / adjudication口径 in src/guardrails.py (isolation, prominence,
sub-30 ms). The template + NCC operate on the HP2k -> 1 ms RMS ENERGY ENVELOPE
(dsp.hp_rms_envelope, the canonical §5.1 amplitude surface), NOT the bipolar HP2k
waveform. A mouse click is a mechanical transient: its fine oscillation phase is
random click-to-click, so a median of the bipolar waveform washes out — measured
template self-NCC (clean calibration clicks vs their own median template) was only
~0.21–0.37 bipolar vs ~0.92–0.94 on the energy envelope, which IS stable across
clicks (the ~5–15 ms rise/decay shape). The envelope domain was chosen from that
measurement (the template-quality self-check below is what surfaced it).

Products (verify_report):
  labels/<name>.verified.csv        gitignored, regeneratable, same 4-col schema as
                                    seed.csv: accept(确定) + adjudicate(存疑) downs +
                                    up rows passed through; reject rows dropped.
  labels/<name>.verify-review.csv   COMMITTED human-adjudication queue (SV-loadable):
                                    adjudicate + reject downs with decision/reason/
                                    ncc/range_db (audit trail in version control).
  docs/verify-report.md             COMMITTED: template quality, score distribution,
                                    threshold basis, per-file accept/reject/裁决 counts.
  figures/<name>_ncc_hist.png       gitignored: histogram + fitted Gaussians + θ lines.

Usage:
    python src/verify.py             # all 4 files -> CSVs + report + figures
    python src/verify.py BJ_C3       # one file (no pooled-θ fallback)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from scipy.ndimage import gaussian_filter1d

import dsp
import export_seeds as es
import guardrails as gr
import snap

REPO = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO / "labels"
FIG_DIR = REPO / "figures"
DOCS_DIR = REPO / "docs"
AUDIT_FILES = gr.AUDIT_FILES                 # ("BJ_C1", "BJ_C3", "F67", "IntlSB20")
SR = dsp.SR

# --- alignment / scoring window (annotation口径) -------------------------------
PRE_MS, POST_MS = 2.0, 15.0          # window [-2, +15] ms around the snapped onset
LAG_MS = 2.0                         # NCC lag search ±2 ms (anchor jitter tolerance)
PRE = round(PRE_MS * SR / 1000.0)    # 96
POST = round(POST_MS * SR / 1000.0)  # 720
LAG = round(LAG_MS * SR / 1000.0)    # 96
SEG_LEN = PRE + POST + 1             # 817 — both endpoints inclusive (avoid off-by-one)

# --- subset / decision constants ----------------------------------------------
ISO_MS = 120.0                       # isolation口径 (matches guardrails._isolated_mask)
MIN_PROM_DB = snap.MIN_PROMINENCE_DB         # 12.0 — reuse, do not redefine
# Physical lower bound for two DISTINCT down presses. MIN_PEAK_GAP_MS=30 is a peak-
# level merge floor (protocol §5 / export_seeds), NOT this. Protocol names 30–50 ms
# as the "safe" down->down band; we take the upper end as the floor below which two
# downs cannot be two physical clicks. 50 > 30 ⇒ within a sub-30 ms close pair the
# "real double-click, keep both" branch never fires (self-consistent: <30 ms is one
# click split). Kept as a named constant so the rule is self-documenting / extensible.
DOWN_MIN_GAP_MS = 50.0

# --- threshold derivation -----------------------------------------------------
POST_LO, POST_HI = 0.1, 0.9          # doubt-band posterior cutoffs: P(click|s)=0.1/0.9
SEPARATION_MIN = 1.0                  # bimodality guard |μ_hi-μ_lo|/(σ_hi+σ_lo)
WEIGHT_MIN = 0.05                     # a component fitting <5% of data isn't a real mode
THETA_FALLBACK_LO, THETA_FALLBACK_HI = 0.5, 0.85   # conservative θ for un-bimodal files
GMM_MIN_N = 8                         # below this, GMM is meaningless -> fallback


# A per-`down` candidate is a mutable dict (dedup rewrites decision/reason in place):
#   sample:int       snapped down onset sample
#   range_db:float   snap_v2 pre-peak prominence (the hard accept gate)
#   iso:bool         isolated >=120 ms from both neighbours (calibration-subset member)
#   ncc:float        peak zero-mean NCC vs the file template ∈[-1,1]
#   decision:str     "确定" (accept) | "drop" (reject) | "存疑" (adjudicate)
#   reason:str       ncc_hi / ncc_lo / mid_band / close_pair_low / both_high_too_close /…
#   clipped:bool     window ran off a file edge (zero-padded)
#   zero_norm:bool   candidate window was silent


# =============================================================================
# A. segment extraction + L2 + NCC
# =============================================================================

def extract_segment(sig: np.ndarray, onset: int, pre: int = PRE,
                    post: int = POST) -> np.ndarray:
    """sig[onset-pre : onset+post+1] as float64, zero-padded off either edge.

    ``sig`` is the aligned 1-D surface (production: the HP2k 1 ms RMS envelope).
    Always returns length ``pre+post+1`` so every segment is dimensionally
    consistent (zero-pad biases a clipped click's NCC slightly low — conservative).
    """
    n = len(sig)
    lo, hi = onset - pre, onset + post + 1
    seg = np.zeros(pre + post + 1, dtype=np.float64)
    a, b = max(lo, 0), min(hi, n)
    if b > a:
        seg[a - lo:b - lo] = sig[a:b]
    return seg


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """Unit-L2 vector, or None for a (near-)zero-norm segment (silence)."""
    nrm = float(np.linalg.norm(v))
    if nrm <= eps:
        return None
    return v / nrm


def build_template(sig: np.ndarray, onsets) -> tuple[np.ndarray | None, int]:
    """Per-sample MEDIAN of the L2-normalized calibration segments, then ZERO-MEANED.

    Median (not mean) is robust to the handful of contaminated segments in the
    not-yet-human-confirmed calibration subset. The template is then mean-subtracted
    and unit-normalized so scoring is a true normalized cross-correlation (Pearson r,
    ∈[-1,1]) on the energy-SHAPE — mean-centering removes the non-negative envelope's
    DC pedestal, which plain cosine cannot (flat noise scores ~0.7 by cosine, ~0 by
    Pearson). Returns (zero-mean unit template, n_used) or (None, 0).
    """
    segs = [u for u in (l2_normalize(extract_segment(sig, int(o))) for o in onsets)
            if u is not None]
    if not segs:
        return None, 0
    med = np.median(np.array(segs), axis=0)
    tmpl = l2_normalize(med - med.mean())
    if tmpl is None:
        return None, 0
    return tmpl, len(segs)


def ncc_score(sig: np.ndarray, onset: int, tmpl: np.ndarray,
              pre: int = PRE, post: int = POST, lag: int = LAG) -> float:
    """Peak ZERO-MEAN normalized cross-correlation (Pearson r) vs the template, ±lag.

    Cut a window widened by ±lag, slide the SEG_LEN sub-window over the 2*lag+1
    integer offsets, mean-subtract each shifted slice and unit-normalize it, dot with
    the zero-mean unit template, take the max. Each term is a Pearson correlation of
    two centered unit vectors ∈[-1,1], so the max ∈[-1,1]. Centering measures SHAPE
    independent of the envelope's DC offset. Flat/silent slices are skipped;
    all-flat -> 0.0.
    """
    big = extract_segment(sig, int(onset), pre + lag, post + lag)
    length = pre + post + 1
    best = -1.0
    found = False
    for d in range(2 * lag + 1):
        c = big[d:d + length]
        cc = c - c.mean()                  # zero-mean: Pearson r at this lag
        nrm = float(np.linalg.norm(cc))
        if nrm <= 1e-12:
            continue
        s = float(cc @ tmpl) / nrm         # tmpl is zero-mean unit-norm, ||tmpl|| omitted
        if s > best:
            best = s
        found = True
    return best if found else 0.0


# =============================================================================
# B. data-driven thresholds — 2-component 1-D Gaussian mixture (no sklearn)
# =============================================================================

def _gauss(x, mu: float, var: float):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def fit_gmm_1d(scores: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> dict:
    """Deterministic 2-component 1-D GMM via EM. Init from score quartiles (no RNG).

    Returns {mu, var, w, loglik, n_iter} with components SORTED by mu ascending
    (index 0 = non-click / low, 1 = click / high). Variances floored to avoid
    collapse onto a duplicated score (many candidates may sit at exactly 0.0 / 1.0).
    """
    s = np.asarray(scores, dtype=np.float64)
    q = np.quantile(s, [0.25, 0.75])
    mu = np.array([q[0], q[1]], dtype=np.float64)
    v0 = max(float(s.var()), 1e-4)
    var = np.array([v0, v0])
    w = np.array([0.5, 0.5])
    var_floor = 1e-4
    prev_ll = -np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        p0 = w[0] * _gauss(s, mu[0], var[0])
        p1 = w[1] * _gauss(s, mu[1], var[1])
        denom = p0 + p1 + 1e-300
        g0, g1 = p0 / denom, p1 / denom
        ll = float(np.sum(np.log(denom)))
        n0, n1 = g0.sum() + 1e-12, g1.sum() + 1e-12
        w = np.array([n0, n1]) / len(s)
        mu = np.array([float(np.sum(g0 * s) / n0), float(np.sum(g1 * s) / n1)])
        var = np.array([
            max(float(np.sum(g0 * (s - mu[0]) ** 2) / n0), var_floor),
            max(float(np.sum(g1 * (s - mu[1]) ** 2) / n1), var_floor),
        ])
        if abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll
    order = np.argsort(mu)
    return {"mu": mu[order], "var": var[order], "w": w[order],
            "loglik": prev_ll, "n_iter": n_iter}


def _posterior_click(x, mu, var, w):
    p0 = w[0] * _gauss(x, mu[0], var[0])
    p1 = w[1] * _gauss(x, mu[1], var[1])
    return p1 / (p0 + p1 + 1e-300)


def _first_cross(grid: np.ndarray, post: np.ndarray, target: float) -> float:
    """First grid point where post >= target; grid[-1] if it never reaches target."""
    idx = int(np.argmax(post >= target))
    if post[idx] < target:
        return float(grid[-1])
    return float(grid[idx])


def histogram_valley(scores: np.ndarray, bins: int = 50, sigma: float = 2.0) -> float:
    """Smoothed-histogram antimode between the two dominant modes (B2 diagnostic).

    Used only as a cross-check / report number, NOT the operative threshold. Returns
    nan if fewer than two modes are found.
    """
    s = np.asarray(scores, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return float(lo)
    counts, edges = np.histogram(s, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    sm = gaussian_filter1d(counts.astype(float), sigma=sigma)
    peaks, _ = sps.find_peaks(sm)
    if len(peaks) < 2:
        return float("nan")
    top2 = sorted(peaks[np.argsort(sm[peaks])[-2:]])
    valley = top2[0] + int(np.argmin(sm[top2[0]:top2[1] + 1]))
    return float(centers[valley])


def derive_thresholds(scores, pooled=None) -> dict:
    """θ_lo / θ / θ_hi from the score distribution's GMM posterior, per file.

    θ = P(click|s)=0.5 (valley); θ_lo/θ_hi = P=0.1/0.9 (the doubt band). If the file
    is NOT credibly bimodal (separation, weight, or no interior 0.1↔0.9 crossing),
    DON'T invent a valley: try the pooled-corpus distribution, else fall back to a
    flagged conservative default. `basis` records which path was taken.
    """
    s = np.asarray(scores, dtype=np.float64)
    out = {"theta": float("nan"), "theta_lo": float("nan"), "theta_hi": float("nan"),
           "bimodal": False, "separation": float("nan"), "basis": "",
           "gmm": None, "hist_valley": float("nan")}
    if len(s) < GMM_MIN_N or float(np.ptp(s)) < 1e-9:
        out.update(theta_lo=THETA_FALLBACK_LO, theta_hi=THETA_FALLBACK_HI,
                   theta=0.5 * (THETA_FALLBACK_LO + THETA_FALLBACK_HI),
                   basis="fallback:degenerate")
        return out

    gmm = fit_gmm_1d(s)
    out["gmm"] = gmm
    out["hist_valley"] = histogram_valley(s)
    mu, var, w = gmm["mu"], gmm["var"], gmm["w"]
    sd = np.sqrt(var)
    sep = float(abs(mu[1] - mu[0]) / (sd[0] + sd[1] + 1e-12))
    out["separation"] = sep

    grid = np.linspace(float(s.min()), float(s.max()), 2000)
    post = _posterior_click(grid, mu, var, w)
    crosses = bool(post.min() <= POST_LO and post.max() >= POST_HI)
    if sep >= SEPARATION_MIN and float(min(w)) >= WEIGHT_MIN and crosses:
        clamp = lambda v: float(min(max(v, mu[0]), mu[1]))
        out.update(theta_lo=clamp(_first_cross(grid, post, POST_LO)),
                   theta=clamp(_first_cross(grid, post, 0.5)),
                   theta_hi=clamp(_first_cross(grid, post, POST_HI)),
                   bimodal=True, basis="gmm")
        return out

    # not bimodal on this file: try the pooled corpus before a fixed constant
    if pooled is not None and len(np.asarray(pooled)) >= GMM_MIN_N:
        pr = derive_thresholds(pooled)            # pooled=None inside -> no recursion loop
        if pr["bimodal"]:
            out.update(theta_lo=pr["theta_lo"], theta=pr["theta"],
                       theta_hi=pr["theta_hi"], bimodal=False, basis="gmm:pooled")
            return out
    out.update(theta_lo=THETA_FALLBACK_LO, theta_hi=THETA_FALLBACK_HI,
               theta=0.5 * (THETA_FALLBACK_LO + THETA_FALLBACK_HI),
               basis="fallback:unimodal")
    return out


# =============================================================================
# C. three-tier decision + close-pair dedup
# =============================================================================

def classify(ncc: float, range_db: float, theta_lo: float, theta_hi: float,
             min_prom_db: float = MIN_PROM_DB) -> tuple[str, str]:
    """Three-tier (step 5). Prominence is a HARD accept gate: a high-NCC but
    low-prominence candidate is NOT auto-accepted — it falls to adjudication."""
    if ncc < theta_lo:
        return "drop", "ncc_lo"
    if ncc >= theta_hi:
        if range_db >= min_prom_db:
            return "确定", "ncc_hi"
        return "存疑", "ncc_hi_low_prom"
    return "存疑", "mid_band"


def _resolve_cluster(cl: list[dict], theta_hi: float, theta_lo: float,
                     sr: int, real_gap_ms: float) -> None:
    """Mutate decisions for one close cluster (>=2 downs within the flag gap)."""
    highs = [d for d in cl if d["ncc"] >= theta_hi]
    lows = [d for d in cl if d["ncc"] < theta_lo]
    n = len(cl)
    # exactly one clear high, all others clear low -> keep the high, drop the rest
    if len(highs) == 1 and len(lows) == n - 1:
        for d in cl:
            if d is highs[0]:
                continue                          # keep its tier (确定 / or 存疑 if low-prom)
            d["decision"] = "drop"
            d["reason"] = "close_pair_low" if n == 2 else "close_chain_low"
        return
    # all clear-high -> too close to be distinct presses -> adjudicate (don't auto-keep)
    if len(highs) == n:
        gaps = [cl[i + 1]["sample"] - cl[i]["sample"] for i in range(n - 1)]
        if n == 2 and gaps[0] >= round(real_gap_ms * sr / 1000.0):
            for d in cl:                          # real double-click — keep both, tiers as-is
                d["reason"] = "real_double_click"
            return
        for d in cl:
            d["decision"] = "存疑"
            d["reason"] = "both_high_too_close"
        return
    # all clear-low -> adjudicate (template-atypical real click can hide here; don't auto-drop)
    if len(lows) == n:
        for d in cl:
            d["decision"] = "存疑"
            d["reason"] = "both_low_close"
        return
    # mixed / mid-band involvement -> adjudicate the whole cluster
    for d in cl:
        d["decision"] = "存疑"
        d["reason"] = "close_pair_ambiguous"


def dedup_close_downs(downs: list[dict], sr: int, theta_hi: float, theta_lo: float,
                      flag_gap_ms: float = es.MIN_PEAK_GAP_MS,
                      real_gap_ms: float = DOWN_MIN_GAP_MS):
    """Resolve sub-flag_gap_ms down clusters (step 5 dedup). Mutates `decision`/
    `reason` in close clusters, then returns (kept, dropped, adjudicate) partitioned
    by FINAL decision. Singletons pass through with their classify tier untouched.
    Handles chains of >2 (only auto-resolves the unambiguous one-high/rest-low case).
    """
    items = sorted(downs, key=lambda d: d["sample"])
    gap = round(flag_gap_ms * sr / 1000.0)
    clusters, cur = [], []
    for d in items:
        if cur and d["sample"] - cur[-1]["sample"] < gap:
            cur.append(d)
        else:
            if cur:
                clusters.append(cur)
            cur = [d]
    if cur:
        clusters.append(cur)
    for cl in clusters:
        if len(cl) > 1:
            _resolve_cluster(cl, theta_hi, theta_lo, sr, real_gap_ms)
    kept = [d for d in items if d["decision"] == "确定"]
    dropped = [d for d in items if d["decision"] == "drop"]
    adjudicate = [d for d in items if d["decision"] == "存疑"]
    return kept, dropped, adjudicate


# =============================================================================
# IO wrappers
# =============================================================================

def _seed_rows(name: str) -> list[dict]:
    """All rows (sorted by sample) from labels/<name>.seed.csv, with confidence."""
    path = LABELS_DIR / f"{name}.seed.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["sample"]))


def _score_file(name: str) -> dict:
    """Heavy pass: load, HP2k, snap_v2, build template, NCC-score every down.

    Returns a bundle (no thresholds yet, so verify_report can pool scores first).
    """
    # env = HP2k -> 1 ms RMS energy envelope (canonical §5.1 surface). NCC runs on
    # THIS, not the bipolar HP2k waveform — clicks' carrier phase is incoherent
    # click-to-click, so a bipolar template washes out (self-NCC ~0.26 vs ~0.93).
    y, sr, env = gr._load_env(name)
    rows = _seed_rows(name)
    marks = [int(r["sample"]) for r in rows]
    types = [r["type"] for r in rows]
    confs = [r.get("confidence", "确定") for r in rows]
    v2 = snap.snap_v2_marks(env, marks, sr)
    iso = gr._isolated_mask(marks, sr)
    n = len(env)

    downs, up_rows = [], []
    for i, typ in enumerate(types):
        r = v2[i]
        if typ == "down":
            downs.append({
                "sample": int(r.sample), "range_db": float(r.range_db),
                "iso": bool(iso[i]),
                "clipped": bool(r.sample - PRE < 0 or r.sample + POST + 1 > n),
            })
        else:
            up_rows.append((int(r.sample), "up", confs[i]))

    calib = [d["sample"] for d in downs if d["iso"] and d["range_db"] >= MIN_PROM_DB]
    tmpl, n_tmpl = build_template(env, calib)
    if tmpl is not None:
        for d in downs:
            seg = extract_segment(env, d["sample"])
            d["zero_norm"] = bool(l2_normalize(seg) is None)
            d["ncc"] = ncc_score(env, d["sample"], tmpl)
        self_ncc = [ncc_score(env, s, tmpl) for s in calib]
        self_med = float(np.median(self_ncc)) if self_ncc else float("nan")
    else:
        for d in downs:
            d["zero_norm"] = True
            d["ncc"] = 0.0
        self_med = float("nan")

    scores = np.asarray([d["ncc"] for d in downs], dtype=np.float64)
    return {"name": name, "sr": sr, "downs": downs, "up_rows": up_rows,
            "tmpl": tmpl, "n_tmpl": n_tmpl, "self_med": self_med,
            "scores": scores, "n_up": len(up_rows)}


def _finalize(bundle: dict, pooled=None) -> dict:
    """Light pass: derive thresholds, classify, dedup, plot, write per-file CSVs."""
    name, sr, downs = bundle["name"], bundle["sr"], bundle["downs"]
    scores = bundle["scores"]

    if bundle["tmpl"] is None:
        for d in downs:                            # cannot template -> all to裁决
            d["decision"], d["reason"] = "存疑", "no_template"
        thr = {"theta": float("nan"), "theta_lo": float("nan"),
               "theta_hi": float("nan"), "bimodal": False,
               "separation": float("nan"), "basis": "no_template",
               "gmm": None, "hist_valley": float("nan")}
    else:
        thr = derive_thresholds(scores, pooled=pooled)
        for d in downs:
            d["decision"], d["reason"] = classify(
                d["ncc"], d["range_db"], thr["theta_lo"], thr["theta_hi"])
        dedup_close_downs(downs, sr, thr["theta_hi"], thr["theta_lo"])

    _plot_hist(name, scores, thr, bundle["self_med"], bundle["n_tmpl"])
    verified = _write_verified(name, downs, bundle["up_rows"], sr)
    review = _write_review(name, downs, sr)

    n_acc = sum(1 for d in downs if d["decision"] == "确定")
    n_drop = sum(1 for d in downs if d["decision"] == "drop")
    n_adj = sum(1 for d in downs if d["decision"] == "存疑")
    rec = {"name": name, "n_down": len(downs), "n_up": bundle["n_up"],
           "n_tmpl": bundle["n_tmpl"], "self_med": bundle["self_med"],
           "n_accept": n_acc, "n_drop": n_drop, "n_adj": n_adj,
           "verified": verified, "review": review, **{k: thr[k] for k in (
               "theta_lo", "theta", "theta_hi", "bimodal", "separation",
               "basis", "hist_valley")}}
    print(f"[verify] {name}: n_down={len(downs)} n_tmpl={bundle['n_tmpl']} "
          f"self-NCC={bundle['self_med']:.3f} | θ=[{_f(thr['theta_lo'])},"
          f"{_f(thr['theta'])},{_f(thr['theta_hi'])}] {thr['basis']} | "
          f"accept={n_acc} 存疑={n_adj} drop={n_drop}")
    return rec


def _f(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def _plot_hist(name: str, scores: np.ndarray, thr: dict, self_med: float,
               n_tmpl: int) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if len(scores):
        lo = min(0.0, float(scores.min()))
        ax.hist(scores, bins=50, range=(lo, 1.0), color="steelblue",
                alpha=0.8, density=True)
    gmm = thr.get("gmm")
    if gmm is not None:
        xs = np.linspace(-0.2, 1.0, 500)
        for k, c in ((0, "salmon"), (1, "darkgreen")):
            ax.plot(xs, gmm["w"][k] * _gauss(xs, gmm["mu"][k], gmm["var"][k]),
                    color=c, lw=1.0)
    for v, col, lab in ((thr["theta_lo"], "orange", "θ_lo"),
                        (thr["theta"], "red", "θ"),
                        (thr["theta_hi"], "green", "θ_hi")):
        if v == v:
            ax.axvline(v, color=col, lw=1.0, ls="--", label=f"{lab}={v:.3f}")
    ax.legend(fontsize=7)
    tag = "bimodal" if thr["bimodal"] else thr["basis"]
    ax.set(xlabel="NCC", ylabel="density",
           title=f"{name}: NCC dist  n_tmpl={n_tmpl}  self-NCC={self_med:.3f}  [{tag}]")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"{name}_ncc_hist.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _write_verified(name: str, downs: list[dict], up_rows, sr: int) -> Path:
    """gitignored, 4-col seed schema: accept(确定)+adjudicate(存疑) downs + up rows."""
    out = LABELS_DIR / f"{name}.verified.csv"
    rows = [(d["sample"], "down", d["decision"]) for d in downs
            if d["decision"] in ("确定", "存疑")]
    rows += [(s, typ, conf) for s, typ, conf in up_rows]
    rows.sort(key=lambda r: r[0])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_s", "type", "confidence"])
        for s, typ, conf in rows:
            w.writerow([s, f"{s / sr:.6f}", typ, conf])
    return out


def _write_review(name: str, downs: list[dict], sr: int) -> Path:
    """COMMITTED adjudication queue (SV-loadable): adjudicate(存疑)+reject(drop) downs."""
    out = LABELS_DIR / f"{name}.verify-review.csv"
    rows = sorted((d for d in downs if d["decision"] in ("存疑", "drop")),
                  key=lambda d: d["sample"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_s", "type", "decision", "reason", "ncc", "range_db"])
        for d in rows:
            w.writerow([d["sample"], f"{d['sample'] / sr:.6f}", "down",
                        d["decision"], d["reason"], f"{d['ncc']:.4f}",
                        f"{d['range_db']:.2f}"])
    return out


def verify_one(name: str) -> dict:
    """Score + finalize one file (no pooled-θ fallback for the single-file path)."""
    return _finalize(_score_file(name))


def verify_report(name: str | None = None) -> Path:
    """Run all 4 files; write committed docs/verify-report.md + per-file CSVs.

    `name` is accepted for CLI-dispatch uniformity but ignored — this is a whole-
    corpus document regenerated (never hand-edited) from the measurement. Scores are
    pooled across files so a file that isn't individually bimodal can borrow the
    corpus valley before falling back to a flagged conservative default.
    """
    bundles = {nm: _score_file(nm) for nm in AUDIT_FILES}
    pool = [b["scores"] for b in bundles.values() if len(b["scores"])]
    pooled = np.concatenate(pool) if pool else None
    recs = [_finalize(bundles[nm], pooled=pooled) for nm in AUDIT_FILES]
    return _write_report(recs)


def _write_report(recs: list[dict]) -> Path:
    tot_down = sum(r["n_down"] for r in recs)
    tot_acc = sum(r["n_accept"] for r in recs)
    tot_drop = sum(r["n_drop"] for r in recs)
    tot_adj = sum(r["n_adj"] for r in recs)
    tot_tmpl = sum(r["n_tmpl"] for r in recs)

    q_rows = "\n".join(
        f"| {r['name']} | {r['n_tmpl']} | {r['self_med']:.3f} | "
        f"{'极性/形状一致' if r['self_med'] >= 0.7 else '⚠ 偏低，查极性/方差'} |"
        for r in recs)
    t_rows = "\n".join(
        f"| {r['name']} | {r['n_down']} | {_f(r['theta_lo'])} | {_f(r['theta'])} | "
        f"{_f(r['theta_hi'])} | {'✓' if r['bimodal'] else '✗'} | "
        f"{_f(r['separation'])} | {_f(r['hist_valley'])} | `{r['basis']}` |"
        for r in recs)
    d_rows = "\n".join(
        f"| {r['name']} | {r['n_down']} | {r['n_accept']} "
        f"({100 * r['n_accept'] / max(r['n_down'], 1):.1f}%) | {r['n_adj']} | "
        f"{r['n_drop']} | {r['n_up']} |" for r in recs)

    flagged = [r["name"] for r in recs if not r["bimodal"]]
    flag_note = ("> ⚠ **未自动双峰分离的文件：** "
                 + "、".join(f"`{n}`" for n in flagged)
                 + "——阈值取 pooled-θ 或保守缺省，已在 `basis` 列标注，开标前请人工签核其 "
                   "accept/裁决边界。\n" if flagged else
                 "> 全部文件 NCC 分布双峰分离良好，阈值取自各自后验谷底。\n")

    md = f"""# verify（模板匹配）报告 — 流水线第 4 步

**日期：** 2026-06-12
**生成：** `python src/verify.py`（数字、CSV、图均由脚本写入，勿手改——重跑覆盖）
**方法串：** `NCC(零均值/Pearson): HP2k→1msRMS-env / win[-2,+15]ms / L2 / 每文件中位模板(孤立&突出≥12dB) / lag±2ms / GMM后验带(P=0.1/0.5/0.9)`

## 背景

snap_v2 冻结后（`annotation-protocol.md` §3），种子仍是**过检上界**。本步对每个 `down`
候选以其 snap_v2 onset 为锚、从 **HP2k→1 ms RMS 能量包络**（`dsp.hp_rms_envelope`，§5.1
唯一真源幅度面）截 `[-2,+15] ms` 窗、L2 归一，取**每文件自建模板**（孤立 ≥{ISO_MS:g} ms 且
突出度 ≥{MIN_PROM_DB:g} dB 的决策级子集逐样本中位）做归一化互相关（NCC，lag ±{LAG_MS:g} ms）。
阈值**先测量分布、再从 GMM 后验谷底导出**（不拍脑袋）；非双峰文件先借 pooled 语料、再退
保守缺省并显式标注（不编造假谷底）。三档判决沿用存疑带哲学，sub-{es.MIN_PEAK_GAP_MS:g} ms
近距对去重。

> **信号域选择（实测驱动）：** 模板用能量包络而非双极 HP2k 原波形。鼠标点击是机械瞬态，
> 逐次载波相位随机、跨点击不相干——**在同一相关度量（cosine）下**对比：双极原波形中位模板
> 被洗成接近零（自 NCC 仅 0.21–0.37，连干净真点击都不判别），能量包络的 ~5–15 ms 起跳-衰减
> 形状跨点击稳定（自 NCC 0.92–0.94）。据此选包络域。**本步操作度量再改为零均值 NCC**
> （Pearson，去包络 DC 基线、判别力更强；cosine 对非负包络有 ~0.7 高地板）——其自 NCC 见
> §1 表（0.62–0.84，低于 cosine 属预期，因去掉了 DC 抬升）。决策由下方"模板质量"自检暴露。

## 1. 模板质量（自校准子集的自 NCC 中位）

| file | n_tmpl | self-NCC 中位 | 判读 |
|---|--:|--:|---|
{q_rows}
| **合计** | {tot_tmpl} | — | — |

> 自 NCC 中位 = 校准子集各候选对本文件模板的 NCC 中位数。高（≥0.7）说明能量包络形状跨
> 点击一致、模板可信；偏低则提示形状方差大或 onset 对齐不准，需复查（包络域已是当前最优，
> 实测优于双极原波形与 `abs()` 整流）。

## 2. 得分分布与阈值依据（每文件后验带）

| file | n_down | θ_lo (P=0.1) | θ (P=0.5) | θ_hi (P=0.9) | 双峰 | 分离度 | 直方谷底(校验) | basis |
|---|--:|--:|--:|--:|:-:|--:|--:|---|
{t_rows}

- θ_lo/θ/θ_hi = GMM 后验 `P(click|s)` 取 0.1/0.5/0.9 处；钳于两分量均值间。
- 分离度 = `|μ_hi-μ_lo|/(σ_hi+σ_lo)`，≥{SEPARATION_MIN:g} 且最小权重 ≥{WEIGHT_MIN:g} 且后验穿过
  0.1↔0.9 才判双峰。直方谷底列为免费交叉校验（应与 θ 接近）。

{flag_note}
## 3. 三档判决计数

| file | n_down | auto-accept(确定) | 中间裁决(存疑) | auto-reject(drop) | up透传 |
|---|--:|--:|--:|--:|--:|
{d_rows}
| **合计** | {tot_down} | {tot_acc} ({100 * tot_acc / max(tot_down, 1):.1f}%) | {tot_adj} | {tot_drop} | — |

- **接受门（不对称）：** `NCC ≥ θ_hi 且 range_db ≥ {MIN_PROM_DB:g} dB` 才 auto-accept；高 NCC 但
  低突出度落裁决（突出度是硬门）。`NCC < θ_lo` → auto-reject；中间带 → 裁决。
- **近距对去重（sub-{es.MIN_PEAK_GAP_MS:g} ms）：** 一高一低→留高弃低；双高（<{DOWN_MIN_GAP_MS:g} ms 物理下限
  内）→裁决；双低/涉中间带→裁决；不自动留双、不自动删真 down。

## 4. 产物

- `labels/<name>.verified.csv` —— **gitignored、可重生成**（同 seed.csv 四列）：accept(确定)
  + 裁决带(存疑) down + up 原样透传；reject 丢弃。
- `labels/<name>.verify-review.csv` —— **committed** 人工裁决队列（SV 可加载）：裁决带 + 被拒项，
  带 `decision/reason/ncc/range_db`，留审计痕迹。
- `figures/<name>_ncc_hist.png` —— gitignored，直方图 + 双高斯 + θ 竖线。

> 方法学注：模板用 **HP2k→1 ms RMS 能量包络**（非双极原波形——载波相位逐次随机、跨点击
> 不相干；同 cosine 度量下原波形自 NCC ~0.2–0.4、包络 ~0.9），操作度量用**零均值 NCC**
> （Pearson，自 NCC 见 §1）。阈值数据驱动、确定性初始化可复现。固化逻辑测试见
> `tests/test_verify.py`。裁决队列受版本控制（同 seed.csv 被 gitignore→committed review CSV 的口径）。
"""
    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "verify-report.md"
    out.write_text(md, encoding="utf-8")
    print(f"\n[verify] 合计 n_down={tot_down} accept={tot_acc} "
          f"({100 * tot_acc / max(tot_down, 1):.1f}%) 存疑={tot_adj} drop={tot_drop} "
          f"-> {out.relative_to(REPO)}")
    return out


def main(argv: list[str]) -> None:
    if argv:
        verify_one(argv[0])
    else:
        verify_report()


if __name__ == "__main__":
    main(sys.argv[1:])
