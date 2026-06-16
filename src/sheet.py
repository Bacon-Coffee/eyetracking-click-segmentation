"""sheet.py — parse the human coarse-annotation spreadsheet into per-click time windows.

New-batch ingest (CLAUDE.md / README §4 workflow, replacing the Sonic Visualiser
marking step for this batch): the human listened to each recording and noted, in
`Click Record SheetRecords.xlsx`, an APPROXIMATE time window for every click
(`click1..click7`, one row per recording). This module turns those messy cells into
machine windows that `locate.py` can refine to a sample-accurate onset.

Crucially these windows are COARSE (~±1 s, sometimes a single second). They are NOT
ground truth — they say "a click is somewhere around here", nothing more. The DSP
(`locate.py`) does the sample-level work inside the window; anything ambiguous is
flagged for human sign-off, never silently guessed.

xlsx is read with the Python STANDARD LIBRARY (zipfile + xml.etree) on purpose: the
project's conda env (`environment.yml`) is frozen and has no openpyxl/pandas, and an
xlsx is just a zip of XML — no new dependency needed.

Cell formats handled (seen in the real sheet):
    "7s-8s"            second range            -> (7, 8)
    "1'03-1'04"        minute'second range     -> (63, 64)
    "59s-1'00"         crosses the minute      -> (59, 60)
    "4'36"             single minute'second    -> (276, 276)
    "2s" / "33s"       single second           -> (2, 2) / (33, 33)
    "Separate audio"   click is in a QN clip   -> flag=separate_audio (parse no time)
    ""                 missing click           -> flag=missing
    "6'58-'6'59"       stray-quote typo        -> (418, 419) + flag=typo
    "2s-3s(Answering starts ...)"  embedded note -> (2, 3) + flag=note (full text kept)
    "1'23-1'24start to answer (1'31-1'32click)"  -> prefers the (... click) time (91,92)

Usage:
    python src/sheet.py                 # dump the parsed sheet (debug)
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
XLSX = REPO / "Click Record SheetRecords.xlsx"
NEW_AUDIO_DIR = REPO / "new audio"

# Pilot recordings already processed (committed). They reappear in this batch under
# the SAME sanitized names; per the user's decision we SKIP them so the committed
# pilot products are never clobbered.
SKIP_STEMS = {"BJ_C1", "BJ_C3", "F67", "IntlSB20"}

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# A time token is either   <min>'<sec>   or   <sec>s   (both self-describing in the
# real sheet, so range halves never need a shared unit).
_MINSEC = re.compile(r"(\d+)'(\d+)")
_SEC = re.compile(r"(\d+)\s*s\b", re.IGNORECASE)
_PAREN_CLICK = re.compile(r"\(([^)]*click[^)]*)\)", re.IGNORECASE)


def wav_stem(name: str) -> str:
    """Sanitize an xlsx row name / filename to the decode.safe_output_name stem.

    Mirrors decode.safe_output_name (keep [A-Za-z0-9], collapse the rest to '_'),
    so a sheet row 'HZ F7' maps to the decoded 'HZ_F7.wav'.
    """
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s or "track"


def parse_qn(stem: str) -> tuple[str, int] | None:
    """A QN-variant stem -> (base_stem, q_index), else None.

    '25REF203Q1' -> ('25REF203', 1); 'HZ_F14Q2' -> ('HZ_F14', 2). The Q index is the
    human's click number, but note it does NOT always equal the 'Separate audio' cell
    position (see HZ F14) — locate.py treats the clip as single-click and flags the
    slot for the human, so an off-by-one Q index does not corrupt the localization.
    """
    m = re.fullmatch(r"(.+?)Q(\d+)", stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _tok_seconds(text: str) -> list[float]:
    """All time tokens in `text`, left to right, as seconds. min'sec before bare sec
    so '1'03' is read as 63 s (not 1 s + 3 s)."""
    minsec_spans = [(m.start(), m.end()) for m in _MINSEC.finditer(text)]
    out: list[tuple[int, float]] = []
    for m in _MINSEC.finditer(text):
        out.append((m.start(), int(m.group(1)) * 60 + int(m.group(2))))
    for m in _SEC.finditer(text):                       # bare <sec>s, skip min'sec overlaps
        if any(a <= m.start() < b for a, b in minsec_spans):
            continue
        out.append((m.start(), float(int(m.group(1)))))
    out.sort(key=lambda t: t[0])
    return [v for _, v in out]


@dataclass(frozen=True)
class Cell:
    raw: str
    t_lo: float | None
    t_hi: float | None
    note: str
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.t_lo is not None


def parse_cell(text: str) -> Cell:
    """Normalize one spreadsheet cell to (t_lo_s, t_hi_s, note, flags)."""
    raw = (text or "").strip()
    if not raw:
        return Cell(raw, None, None, "", ("missing",))

    flags: list[str] = []
    if "separate" in raw.lower():
        return Cell(raw, None, None, raw, ("separate_audio",))

    # Prefer a parenthesized "(... click ...)" time if present — the human sometimes
    # notes the real click time inside a parenthetical after a "start to answer" lead.
    src = raw
    m = _PAREN_CLICK.search(raw)
    if m and _tok_seconds(m.group(1)):
        src = m.group(1)
        flags.append("note")

    secs = _tok_seconds(src)
    # note = any leftover alphabetic content after removing every time token + punctuation
    # (so a clean "7s-8s" / "59s-1'00" is NOT flagged, but "4s starts" / "(repeated)" is).
    leftover = _SEC.sub(" ", _MINSEC.sub(" ", raw))
    leftover = re.sub(r"[\d\s\-'():.,]", "", leftover)
    if re.search(r"[A-Za-z]", leftover) and "note" not in flags:
        flags.append("note")
    if re.search(r"-\s*'", raw):                        # stray-quote typo like 6'58-'6'59
        flags.append("typo")

    if not secs:
        flags.append("unparseable")
        return Cell(raw, None, None, raw, tuple(flags))

    lo = secs[0]
    hi = secs[1] if len(secs) >= 2 else secs[0]
    if hi < lo:                                         # defensive: keep ascending
        lo, hi = hi, lo
    note = raw if flags else ""
    return Cell(raw, float(lo), float(hi), note, tuple(flags))


@dataclass
class Row:
    name: str                 # raw xlsx 'Sample' cell, e.g. "25REF203"
    wav_stem: str             # sanitized stem, e.g. "25REF203"
    cells: list[Cell]         # one per non-empty click column (up to 7)
    anomalies: list[str] = field(default_factory=list)


def _read_grid(path: Path = XLSX) -> list[list[str]]:
    """xlsx sheet1 -> list of rows of cell strings (stdlib, no openpyxl)."""
    z = zipfile.ZipFile(path)
    ss: list[str] = []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    for si in root.findall(f"{_NS}si"):
        ss.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))

    def colnum(ref: str) -> int:
        letters = re.match(r"([A-Z]+)", ref).group(1)
        c = 0
        for ch in letters:
            c = c * 26 + (ord(ch) - 64)
        return c - 1

    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows: dict[int, dict[int, str]] = {}
    maxc = 0
    for row in sheet.iter(f"{_NS}row"):
        ri = int(row.attrib["r"])
        for c in row.findall(f"{_NS}c"):
            ci = colnum(c.attrib["r"])
            maxc = max(maxc, ci)
            t = c.attrib.get("t")
            v = c.find(f"{_NS}v")
            val = ""
            if v is not None:
                val = ss[int(v.text)] if t == "s" else (v.text or "")
            elif c.find(f"{_NS}is") is not None:
                val = "".join(tt.text or "" for tt in c.iter(f"{_NS}t"))
            rows.setdefault(ri, {})[ci] = val.strip()
    out = []
    for ri in sorted(rows):
        out.append([rows[ri].get(ci, "") for ci in range(maxc + 1)])
    return out


def read_sheet(path: Path = XLSX, skip: set[str] = SKIP_STEMS) -> list[Row]:
    """Parse the spreadsheet into Row objects (header skipped, empty rows dropped).

    Each Row keeps the click cells parsed to windows, plus detected anomalies
    (out-of-order click times, fewer than the typical 7 clicks). Rows whose sanitized
    stem is in `skip` (the committed pilot recordings) are omitted.
    """
    grid = _read_grid(path)
    out: list[Row] = []
    for raw_row in grid[1:]:                            # row 0 is the header
        name = (raw_row[0] if raw_row else "").strip()
        if not name:
            continue
        stem = wav_stem(name)
        if stem in skip:
            continue
        cells = [parse_cell(c) for c in raw_row[1:8]]   # click1..click7
        while cells and not cells[-1].raw:              # drop trailing empty columns
            cells.pop()

        anomalies: list[str] = []
        n_real = sum(1 for c in cells if c.raw)
        if n_real and n_real < 7:
            anomalies.append(f"only_{n_real}_clicks")
        los = [c.t_lo for c in cells if c.t_lo is not None]
        if any(b < a for a, b in zip(los, los[1:])):
            anomalies.append("out_of_order")
        out.append(Row(name=name, wav_stem=stem, cells=cells, anomalies=anomalies))
    return out


def main() -> None:
    rows = read_sheet()
    print(f"{len(rows)} recordings (pilot {sorted(SKIP_STEMS)} skipped)\n")
    for r in rows:
        tag = f"  [{','.join(r.anomalies)}]" if r.anomalies else ""
        print(f"{r.name:16s} -> {r.wav_stem}{tag}")
        for i, c in enumerate(r.cells, 1):
            win = f"{c.t_lo:.0f}-{c.t_hi:.0f}s" if c.ok else "—"
            fl = f"  ({','.join(c.flags)})" if c.flags else ""
            print(f"    click{i}: {win:12s} {c.raw!r}{fl}")


if __name__ == "__main__":
    main()
