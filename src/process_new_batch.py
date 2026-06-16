"""process_new_batch.py — end-to-end driver for the xlsx-annotated new-audio batch.

Pipeline for the 54 new recordings (pilot 4 skipped) + their 7 QN separate-audio clips:

    decode (m4a -> 48k mono f32 WAV)            reuse decode.decode_one
      -> build coarse windows from the xlsx     reuse sheet.read_sheet
      -> self-calibrate a batch click template  reuse locate.template_segments
      -> localize one onset per window          reuse locate.locate_file (snap_v2 + NCC)
      -> write located.csv + cuts.csv           locate.write_*  (committed)
      -> cut PCM into segments                   reuse cut.cut_file
      -> overlay figure + review queue + report  for human sign-off

The xlsx windows are coarse (~±1 s), so this is "best estimate + flag for review",
NOT ground truth. Strict mir_eval onset F-measure is NOT computed here (no fine GT for
this batch); the report is a coverage/confidence summary and the overlays are the
sign-off surface. We never touch the committed pilot products (different stems; we call
cut.cut_file per stem, not cut.main(), and write our own report).

Usage:
    python src/process_new_batch.py            # decode (if needed) + full batch
    python src/process_new_batch.py --no-decode
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import cut
import decode
import locate
import sheet

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "data" / "wav"
LABELS_DIR = REPO / "labels"
DOCS_DIR = REPO / "docs"
NEW_AUDIO = REPO / "new audio"
REVIEW_CSV = LABELS_DIR / "new-batch-review.csv"
REPORT_MD = DOCS_DIR / "new-batch-report.md"

_REVIEW_COLS = ["file", "click_idx", "window_text", "t_lo_s", "t_hi_s",
                "reason", "sample", "time_s", "ncc", "range_db", "note"]


@dataclass
class Item:
    stem: str                      # decoded wav stem
    m4a: Path
    kind: str                      # 'main' | 'qn'
    windows: list[locate.WindowSpec]
    row_name: str                  # display name of the source recording
    base_stem: str | None = None   # for 'qn': the main recording stem it belongs to


def _csv_field(v) -> str:
    s = "" if v is None else str(v)
    return f'"{s}"' if ("," in s or '"' in s or "\n" in s) else s


def build_items(rows: list[sheet.Row], file_map: dict[str, Path]
                ) -> tuple[list[Item], list[dict], list[str]]:
    """Turn sheet rows + audio files into processable items + sheet-level review rows.

    Returns (items, review_rows, unmatched). Main items carry one window per parsed,
    non-separate click cell. QN clips become their own single-window items. Missing /
    separate / unparseable cells and unmatched recordings go to review_rows.
    """
    items: list[Item] = []
    review: list[dict] = []
    unmatched: list[str] = []

    row_by_stem = {r.wav_stem: r for r in rows}
    stems_with_row = set(row_by_stem)

    for r in rows:
        m4a = file_map.get(r.wav_stem)
        if m4a is None:
            unmatched.append(r.wav_stem)
            review.append({"file": r.wav_stem, "reason": "unmatched",
                           "note": "sheet row has no matching audio file"})
            continue
        for anom in r.anomalies:
            review.append({"file": r.wav_stem, "reason": anom,
                           "note": "; ".join(c.raw for c in r.cells)})
        windows: list[locate.WindowSpec] = []
        for i, c in enumerate(r.cells, start=1):
            if "separate_audio" in c.flags:
                review.append({"file": r.wav_stem, "click_idx": i, "window_text": c.raw,
                               "reason": "separate_audio",
                               "note": "click is in a QN clip; located there"})
                continue
            if "missing" in c.flags:
                review.append({"file": r.wav_stem, "click_idx": i, "reason": "missing",
                               "note": "empty cell"})
                continue
            if not c.ok:
                review.append({"file": r.wav_stem, "click_idx": i, "window_text": c.raw,
                               "reason": "unparseable", "note": c.note})
                continue
            windows.append(locate.WindowSpec(i, c.t_lo, c.t_hi, c.flags))
        items.append(Item(r.wav_stem, m4a, "main", windows, r.name))

    # QN separate-audio clips: their own items, mapped back to a base recording slot.
    for stem, m4a in sorted(file_map.items()):
        qn = sheet.parse_qn(stem)
        if qn is None:
            continue
        base, qidx = qn
        reasons = ["separate_audio"]
        note = f"separate-audio clip for {base} (Q index {qidx})"
        base_row = row_by_stem.get(base)
        if base_row is None:
            note += " — base recording not in sheet/skipped"
        else:
            sep_slots = [i for i, c in enumerate(base_row.cells, 1)
                         if "separate_audio" in c.flags]
            if sep_slots and qidx not in sep_slots:
                reasons.append("qn_slot_ambiguous")
                note += f"; sheet marks Separate audio at click{sep_slots}, not click{qidx}"
        win = locate.WindowSpec(qidx, None, None, tuple(reasons))
        items.append(Item(stem, m4a, "qn", [win], stem, base_stem=base))
        review.append({"file": stem, "click_idx": qidx, "reason": ",".join(reasons),
                       "note": note})
    return items, review, unmatched


def _decode(items: list[Item]) -> None:
    """Decode each item's m4a to data/wav/<stem>.wav (skip if already present)."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (WAV_DIR / f"{it.stem}.wav").exists()]
    print(f"[decode] {len(todo)} new file(s) to decode "
          f"({len(items) - len(todo)} already present)")
    for it in todo:
        try:
            decode.decode_one(it.m4a, WAV_DIR)
        except Exception as e:                       # noqa: BLE001 — keep batch going
            print(f"[decode] FAILED {it.m4a.name}: {e}", file=sys.stderr)


def run(do_decode: bool = True) -> None:
    rows = sheet.read_sheet()
    file_map = {sheet.wav_stem(p.stem): p for p in NEW_AUDIO.glob("*.m4a")
                if sheet.wav_stem(p.stem) not in sheet.SKIP_STEMS}
    items, review, unmatched = build_items(rows, file_map)
    print(f"[batch] {len(items)} items "
          f"({sum(it.kind == 'main' for it in items)} main + "
          f"{sum(it.kind == 'qn' for it in items)} qn); "
          f"{len(unmatched)} unmatched; {len(review)} sheet review rows")

    if do_decode:
        _decode(items)
    items = [it for it in items if (WAV_DIR / f"{it.stem}.wav").exists()]

    # --- pass 1: self-calibrate the batch click template -----------------------
    print("[batch] pass 1/2 — building click template")
    seg_lists: list[list] = []
    for it in items:
        wav = WAV_DIR / f"{it.stem}.wav"
        try:
            env, sr, peaks = locate.load_detect(wav)
            seg_lists.append(locate.template_segments(env, sr, peaks, it.windows))
        except Exception as e:                       # noqa: BLE001
            print(f"[pass1] FAILED {it.stem}: {e}", file=sys.stderr)
    tmpl, n_tmpl = locate.build_global_template(seg_lists)
    print(f"[batch] template from {n_tmpl} prominent first-pass clicks"
          + ("" if tmpl is not None else "  (NONE — falling back to prominence-only)"))

    # --- pass 2: localize, write labels, cut, plot -----------------------------
    print("[batch] pass 2/2 — localizing + cutting")
    recs: list[dict] = []
    for it in items:
        wav = WAV_DIR / f"{it.stem}.wav"
        try:
            env, sr, peaks = locate.load_detect(wav)
            results = locate.locate_file(env, sr, peaks, it.windows, tmpl)
            n = len(env)
            locate.write_located_csv(it.stem, results, sr)
            basis = "sheet-window-qn" if it.kind == "qn" else "sheet-window"
            _, kept, dropped = locate.write_cuts_csv(it.stem, results, sr, n, basis)
            locate.overlay_figure(it.stem, env, sr, results)
            manifest = cut.cut_file(it.stem)         # PCM segments (round-trip asserted)

            n_win = len(results)
            n_loc = sum(r.pick.winner is not None for r in results)
            n_ok = sum(r.pick.confidence == "确定" and not r.pick.reasons for r in results)
            n_flag = n_win - n_ok
            recs.append({"stem": it.stem, "kind": it.kind, "n_win": n_win,
                         "n_loc": n_loc, "n_ok": n_ok, "n_flag": n_flag,
                         "n_seg": len(manifest), "n_dropped": len(dropped)})

            for r in results:                        # per-window review rows
                p = r.pick
                w = p.winner
                if p.confidence == "确定" and not p.reasons:
                    continue
                review.append({
                    "file": it.stem, "click_idx": r.spec.click_idx,
                    "window_text": ("whole-clip" if r.spec.t_lo is None
                                    else f"{r.spec.t_lo:.0f}-{r.spec.t_hi:.0f}s"),
                    "t_lo_s": "" if r.spec.t_lo is None else f"{r.spec.t_lo:.0f}",
                    "t_hi_s": "" if r.spec.t_hi is None else f"{r.spec.t_hi:.0f}",
                    "reason": ",".join(p.reasons) or p.confidence,
                    "sample": "" if w is None else w.onset,
                    "time_s": "" if w is None else f"{w.onset / sr:.6f}",
                    "ncc": "" if w is None else f"{w.ncc:.3f}",
                    "range_db": "" if w is None else f"{w.range_db:.1f}",
                    "note": ",".join(r.spec.sheet_flags),
                })
            print(f"[loc] {it.stem}: {n_loc}/{n_win} located, {n_ok} 确定, "
                  f"{n_flag} flagged -> {len(manifest)} segments")
        except Exception as e:                       # noqa: BLE001
            print(f"[pass2] FAILED {it.stem}: {e}", file=sys.stderr)
            traceback.print_exc()
            review.append({"file": it.stem, "reason": "error", "note": str(e)})

    _write_review(review)
    _write_report(recs, n_tmpl, unmatched, review)


def _write_review(review: list[dict]) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(_REVIEW_COLS) + "\n")
        for d in sorted(review, key=lambda r: (str(r.get("file", "")),
                                               int(r.get("click_idx") or 0))):
            f.write(",".join(_csv_field(d.get(c, "")) for c in _REVIEW_COLS) + "\n")
    print(f"[batch] review queue ({len(review)} rows) -> {REVIEW_CSV.relative_to(REPO)}")


def _write_report(recs: list[dict], n_tmpl: int, unmatched: list[str],
                  review: list[dict]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    main = [r for r in recs if r["kind"] == "main"]
    qn = [r for r in recs if r["kind"] == "qn"]
    tot_win = sum(r["n_win"] for r in recs)
    tot_loc = sum(r["n_loc"] for r in recs)
    tot_ok = sum(r["n_ok"] for r in recs)
    tot_flag = sum(r["n_flag"] for r in recs)
    tot_seg = sum(r["n_seg"] for r in recs)

    body = "\n".join(
        f"| {r['stem']} | {r['kind']} | {r['n_win']} | {r['n_loc']} | {r['n_ok']} | "
        f"{r['n_flag']} | {r['n_seg']} |"
        for r in sorted(recs, key=lambda r: r["stem"]))

    md = f"""# new-batch 处理报告 — xlsx 粗标注驱动定位 + 切割

**生成：** `python src/process_new_batch.py`（脚本写入，勿手改——重跑覆盖）

## 口径与诚实声明

- 输入人工标注 = `Click Record SheetRecords.xlsx`，每 click 一个**粗时间窗（~±1 s）**，
  不是样本级 ground truth。本流程在窗内用 **snap_v2 + 模板 NCC** 定位到样本级 onset，
  对**模糊窗只标记不臆造**（见 `labels/new-batch-review.csv`）。
- 因无样本级 GT，本批**不计算 mir_eval onset F-measure**；下表是**覆盖率/置信度**汇总，
  精度由 `figures/<stem>_locate_overlay.png` **人工签核**。
- 跳过 4 个与试点同名录音（{", ".join(sorted(sheet.SKIP_STEMS))}），committed 试点产物未触动。
- "Separate audio" click 在对应 QN 短片内单独定位（basis=`sheet-window-qn`）。

## 规模

- 录音项：{len(main)} 主 + {len(qn)} QN = {len(recs)} 项；模板取自 {n_tmpl} 个突出首过点击。
- 窗合计 {tot_win}；已定位 {tot_loc}；自动确定（无 flag）{tot_ok}
  （{100 * tot_ok / max(tot_win, 1):.1f}%）；需复核 {tot_flag}；切出片段 {tot_seg}。
- 复核队列 {len(review)} 行；未匹配音频 {len(unmatched)}{(": " + ", ".join(unmatched)) if unmatched else ""}。

## 逐文件

| stem | kind | 窗 | 已定位 | 确定 | 待复核 | 片段 |
|---|---|--:|--:|--:|--:|--:|
{body}

## 人工复核流程

1. 打开 `labels/new-batch-review.csv`，按 `reason` 过滤
   （`low_conf`=匹配/突出度偏弱的弱定位，最该先看/`multi_candidate`=窗内另有一个相距 >150ms 的
   click 样瞬态/`no_candidate`/`out_of_order`/`separate_audio`/`qn_slot_ambiguous`/`typo`/`note`/
   `unmatched`）。
2. 对照 `figures/<stem>_locate_overlay.png` 逐项核对：绿线 onset 是否落在该 click 主瞬态脚点。
3. 错位/漏检 → 手改 `labels/<stem>.cuts.csv`，重跑 `python src/cut.py <stem>` 重切。
"""
    REPORT_MD.write_text(md, encoding="utf-8")
    print(f"[batch] report -> {REPORT_MD.relative_to(REPO)}")
    print(f"[batch] DONE: {tot_loc}/{tot_win} located, {tot_ok} 确定, "
          f"{tot_flag} flagged, {tot_seg} segments")


def main(argv: list[str]) -> None:
    run(do_decode="--no-decode" not in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
