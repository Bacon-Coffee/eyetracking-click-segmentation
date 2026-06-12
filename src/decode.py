"""Decode .m4a recordings to canonical PCM WAV for sample-accurate SED.

Pipeline step 1 (see README §4 / CLAUDE.md): every downstream stage works in the
PCM domain on the output of this module.

Output format (fixed): 48 kHz, mono, float32 WAV.
  - We decode with ffmpeg using explicit codec/rate/channel flags.
  - We DO NOT use `ffmpeg -c copy`: stream copy only cuts/keeps on AAC frame
    boundaries (~21 ms @ 48 kHz) and cannot give sample accuracy.

On AAC priming/encoder delay: AAC prepends a constant number of priming samples.
ffmpeg's decoder compensates for the encoder delay it can read from the
container, but any residual is a *constant* shift of the whole signal. Since both
exploration and the eventual cutting operate on this same decoded WAV, that
constant offset does not affect *inter-click* spacing or relative onset
positions — which is what matters for segmenting one recording into clips.

Filenames contain special / full-width characters (e.g. "！BJ C1.m4a"). We treat
paths literally (never parse the leading prefix) and ASCII-sanitize only the
*output* filename, printing the original->output mapping.

Usage:
    python src/decode.py                      # decode all audio/*.m4a -> data/wav/
    python src/decode.py "audio/% F67.m4a"    # decode one file
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Canonical PCM target — keep in sync with README §4 conventions.
TARGET_SR = 48_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_f32le"  # float32 little-endian PCM

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
WAV_DIR = REPO_ROOT / "data" / "wav"


def ffprobe_stream(path: Path) -> dict:
    """Return the first audio stream's properties (native sr, channels, codec, duration)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    stream = info.get("streams", [{}])[0]
    duration = info.get("format", {}).get("duration")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "duration_s": float(duration) if duration else None,
    }


def safe_output_name(src: Path) -> str:
    """ASCII-sanitize a stem for a filesystem-friendly .wav name.

    Keep [A-Za-z0-9], collapse everything else (spaces, #/%/@/！, full-width) to '_'.
    """
    stem = src.stem
    ascii_stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    if not ascii_stem:  # name was entirely special chars
        ascii_stem = "track"
    return f"{ascii_stem}.wav"


def decode_one(src: Path, dst_dir: Path = WAV_DIR) -> Path:
    """Decode a single m4a to 48 kHz mono float32 WAV. Returns the output path."""
    probe = ffprobe_stream(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / safe_output_name(src)

    upsampling = probe["sample_rate"] and probe["sample_rate"] < TARGET_SR
    note = "  (UPSAMPLING from native)" if upsampling else ""
    print(
        f"[probe] {src.name!r}: codec={probe['codec']} "
        f"sr={probe['sample_rate']} ch={probe['channels']} "
        f"dur={probe['duration_s']:.2f}s{note}"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-c:a", TARGET_CODEC,
        "-sample_fmt", "flt",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    print(f"[decode] {src.name!r} -> {dst.relative_to(REPO_ROOT)}")
    return dst


def decode_all(audio_dir: Path = AUDIO_DIR, dst_dir: Path = WAV_DIR) -> list[Path]:
    sources = sorted(audio_dir.glob("*.m4a"))
    if not sources:
        print(f"No .m4a files found in {audio_dir}", file=sys.stderr)
        return []
    print(f"Decoding {len(sources)} file(s) -> {dst_dir.relative_to(REPO_ROOT)} "
          f"@ {TARGET_SR} Hz mono float32\n")
    return [decode_one(src, dst_dir) for src in sources]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            decode_one(Path(arg))
    else:
        decode_all()
