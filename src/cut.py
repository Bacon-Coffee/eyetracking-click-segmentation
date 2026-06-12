"""cut.py — pipeline step 6: sample-accurate PCM-domain cutting at confirmed marker onsets.

Scope note (2026-06-12 target redefinition): the cut targets are the SEVEN
human-confirmed marker clicks per recording (labels/<name>.cuts.csv, committed,
human-adjudicated single source of truth) — NOT every detected transient.
See docs/cut-report.md for the adjudication trail.

Semantics: each cut point is the snap_v2 onset (down = press transient) of one
marker click. N cut points -> N+1 segments covering the file exactly:
[0, c1), [c1, c2), ..., [cN, EOF). Concatenating all segments reproduces the
input bit-exactly (asserted at runtime).

Usage:
    python src/cut.py            # cut all files that have a cuts.csv
    python src/cut.py BJ_C1 F67  # cut selected files

Output: data/segments/<name>/<name>_seg<k>.wav (gitignored, regenerable)
        + committed docs/cut-report.md (script-written, do not hand-edit).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "data" / "wav"
LABELS_DIR = REPO / "labels"
SEG_DIR = REPO / "data" / "segments"
REPORT = REPO / "docs" / "cut-report.md"

SR = 48_000


def read_cuts(path: Path) -> list[dict]:
    """Read a cuts.csv (sample,time_s,confidence,range_db,basis), sorted by sample."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["sample"]))
    return rows


def segment_bounds(cuts: list[int], n: int) -> list[tuple[int, int]]:
    """N strictly-increasing in-range cut samples -> N+1 [start, end) bounds covering [0, n)."""
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("cut samples must be strictly increasing")
    if cuts and (cuts[0] <= 0 or cuts[-1] >= n):
        raise ValueError("cut samples must lie strictly inside (0, n)")
    edges = [0, *cuts, n]
    return list(zip(edges[:-1], edges[1:]))


def cut_file(name: str) -> list[dict]:
    """Cut one decoded WAV at its confirmed marker onsets. Returns per-segment manifest rows."""
    wav = WAV_DIR / f"{name}.wav"
    info = sf.info(wav)
    y, sr = sf.read(wav, dtype="float32")
    if y.ndim > 1:  # decoded files are mono; guard anyway
        y = y.mean(axis=1)
    rows = read_cuts(LABELS_DIR / f"{name}.cuts.csv")
    cuts = [int(r["sample"]) for r in rows]
    bounds = segment_bounds(cuts, len(y))

    out_dir = SEG_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = 0
    for k, (a, b) in enumerate(bounds):
        seg = y[a:b]  # pure PCM slice — sample-accurate by construction
        out = out_dir / f"{name}_seg{k:02d}.wav"
        sf.write(out, seg, sr, subtype=info.subtype)
        manifest.append({
            "file": name, "seg": k, "start_sample": a, "end_sample": b,
            "start_s": a / sr, "dur_s": (b - a) / sr, "path": out,
        })
        total += len(seg)

    # Invariant: segments partition the input exactly (no lost/duplicated samples).
    assert total == len(y), f"{name}: segment samples {total} != input {len(y)}"
    back = np.concatenate([sf.read(m["path"], dtype="float32")[0] for m in manifest])
    assert len(back) == len(y) and np.array_equal(back, y), f"{name}: round-trip mismatch"
    return manifest


def write_report(all_manifests: dict[str, list[dict]]) -> None:
    lines = [
        "# cut 报告 — 标记点击样本级切割（流水线第 6 步）",
        "",
        "**生成：** `python src/cut.py`（脚本写入，勿手改——重跑覆盖）",
        "",
        "切点 = `labels/<name>.cuts.csv`（committed，人工逐项确认的 7 个标记点击的 snap_v2 down onset；",
        "每文件恰 7 切点 → 8 片段）。切割为 PCM 域纯切片；**逐文件断言**:片段样本数之和 == 原文件、",
        "拼接回读与原文件逐样本相等（bit-exact round-trip）。",
        "",
        "| file | 切点 | 片段 | 片段时长(s) |",
        "|---|--:|--:|---|",
    ]
    for name, man in all_manifests.items():
        durs = " / ".join(f"{m['dur_s']:.2f}" for m in man)
        lines.append(f"| {name} | {len(man) - 1} | {len(man)} | {durs} |")
    lines += [
        "",
        "口径注记:IntlSB20 的 7 个切点为单瞬态确认点（模板精扫未在前方找到统计显著的更早 down，",
        "空白对照 max 0.949 vs 信号窗 0.91–0.977，仅 1 处显著）——若该文件标记同为 down+up 对而 down",
        "低于本底，切点语义相对其他文件可能整体偏晚 ~0.1s 量级，已知限制随报告留档。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem.replace(".cuts", "") for p in LABELS_DIR.glob("*.cuts.csv"))
    all_manifests = {}
    for name in names:
        man = cut_file(name)
        all_manifests[name] = man
        durs = ", ".join("%.1fs" % m["dur_s"] for m in man)
        print(f"[cut] {name}: {len(man) - 1} cuts -> {len(man)} segments ({durs})")
    write_report(all_manifests)
    print(f"[cut] report -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
