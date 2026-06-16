"""Pure-assertion tests for src/sheet.py cell/time parsing. Run: python tests/test_sheet.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sheet import parse_cell, parse_qn, wav_stem  # noqa: E402


def test_second_range():
    c = parse_cell("7s-8s")
    assert (c.t_lo, c.t_hi) == (7.0, 8.0), c
    assert not c.flags, c.flags          # a clean "Ns-Ms" must NOT be flagged as a note


def test_minute_second_range():
    c = parse_cell("1'03-1'04")
    assert (c.t_lo, c.t_hi) == (63.0, 64.0), c
    assert not c.flags


def test_minute_second_single():
    c = parse_cell("4'36")
    assert (c.t_lo, c.t_hi) == (276.0, 276.0)
    assert not c.flags


def test_bare_second_single():
    for txt, sec in (("2s", 2.0), ("33s", 33.0), ("48s", 48.0)):
        c = parse_cell(txt)
        assert (c.t_lo, c.t_hi) == (sec, sec), (txt, c)
        assert not c.flags


def test_minute_crossing():
    c = parse_cell("59s-1'00")
    assert (c.t_lo, c.t_hi) == (59.0, 60.0), c
    assert not c.flags


def test_missing():
    c = parse_cell("")
    assert c.t_lo is None and "missing" in c.flags and not c.ok


def test_separate_audio():
    c = parse_cell("Separate audio")
    assert c.t_lo is None and "separate_audio" in c.flags


def test_typo_stray_quote():
    c = parse_cell("6'58-'6'59")
    assert (c.t_lo, c.t_hi) == (418.0, 419.0), c
    assert "typo" in c.flags


def test_embedded_note_no_click():
    c = parse_cell("2s-3s(Answering starts around 17s-18s with relatively high noise.)")
    assert (c.t_lo, c.t_hi) == (2.0, 3.0), c          # keep the leading window
    assert "note" in c.flags


def test_parenthetical_click_preferred():
    c = parse_cell("1'23-1'24start to answer (1'31-1'32click)")
    assert (c.t_lo, c.t_hi) == (91.0, 92.0), c        # prefer the (... click) time
    assert "note" in c.flags


def test_repeated_note():
    c = parse_cell("2'53(repeated)")
    assert (c.t_lo, c.t_hi) == (173.0, 173.0), c
    assert "note" in c.flags


def test_starts_note():
    c = parse_cell("4s starts")
    assert (c.t_lo, c.t_hi) == (4.0, 4.0), c
    assert "note" in c.flags


def test_parse_qn():
    assert parse_qn("25REF203Q1") == ("25REF203", 1)
    assert parse_qn("HZ_F14Q2") == ("HZ_F14", 2)
    assert parse_qn("C55Q3") == ("C55", 3)
    assert parse_qn("BJ_C14") is None
    assert parse_qn("BJNC3") is None


def test_wav_stem():
    assert wav_stem("HZ F7") == "HZ_F7"
    assert wav_stem("25REF203") == "25REF203"
    assert wav_stem("Wei laoshi-35") == "Wei_laoshi_35"
    assert wav_stem("SM C1") == "SM_C1"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"  ok {fn.__name__}")
    print("all sheet tests passed")
