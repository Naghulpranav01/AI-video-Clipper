from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import static_ffmpeg

if shutil.which("ffmpeg") is None:
    static_ffmpeg.add_paths()
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def run(
    cmd: Sequence[str | Path],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    args = [str(c) for c in cmd]
    proc = subprocess.run(args, capture_output=capture, text=True, cwd=cwd)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{detail[-3000:]}")
    return proc


def probe(path: str | Path) -> dict:
    out = run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {
        "duration": float(data.get("format", {}).get("duration") or 0.0),
        "width": int(video.get("width", 0)) if video else 0,
        "height": int(video.get("height", 0)) if video else 0,
        "fps": _fps(video),
        "has_audio": audio is not None,
        "video_codec": video.get("codec_name") if video else None,
    }


def _fps(video: dict | None) -> float:
    if not video:
        return 0.0
    raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, den = raw.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def run_ffmpeg(
    cmd: Sequence[str | Path],
    *,
    total_seconds: float = 0.0,
    label: str = "ffmpeg",
    cwd: str | Path | None = None,
) -> None:
    """Run ffmpeg and print a progress percentage parsed from -progress output."""
    args = [str(c) for c in cmd]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    assert proc.stdout is not None and proc.stderr is not None
    last_pct = -5
    for line in proc.stdout:
        if not line.startswith("out_time_ms=") or total_seconds <= 0:
            continue
        try:
            seconds = int(line.split("=", 1)[1]) / 1_000_000
        except ValueError:
            continue
        pct = min(100, int(seconds / total_seconds * 100))
        if pct >= last_pct + 10:
            last_pct = pct
            log(f"  {label}: {pct}%")
    stderr = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"{label} failed ({proc.returncode})\n{stderr[-3000:]}")


def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def slugify(text: str, limit: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text or "video")[:limit].strip("-")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dedupe_words(words: Iterable[dict], tolerance: float = 0.35) -> list[dict]:
    """Drop duplicated words produced where overlapping transcription chunks meet."""
    cleaned: list[dict] = []
    for word in sorted(words, key=lambda w: (float(w["start"]), float(w["end"]))):
        if not str(word.get("word", "")).strip():
            continue
        if cleaned:
            prev = cleaned[-1]
            same = str(prev["word"]).strip().lower() == str(word["word"]).strip().lower()
            if same and abs(float(prev["start"]) - float(word["start"])) < tolerance:
                continue
        cleaned.append(
            {
                "word": str(word["word"]).strip(),
                "start": float(word["start"]),
                "end": float(word["end"]),
            }
        )
    return cleaned
