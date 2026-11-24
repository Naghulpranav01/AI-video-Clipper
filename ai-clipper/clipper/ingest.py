from __future__ import annotations

import re
from pathlib import Path

from .config import settings
from .util import FFMPEG, log, probe, run


def is_url(value: str) -> bool:
    return re.match(r"^https?://", value.strip(), re.IGNORECASE) is not None


def fetch_source(value: str, workdir: Path, *, max_height: int | None = None) -> Path:
    """Download a remote video with yt-dlp, or return a local file path as-is."""
    if not is_url(value):
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Input file not found: {path}")
        return path

    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "source.mp4"
    if target.exists() and target.stat().st_size > 0:
        log(f"  using cached download: {target.name}")
        return target

    last = {"pct": -10}

    def hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        if not total:
            return
        pct = int(status.get("downloaded_bytes", 0) / total * 100)
        if pct >= last["pct"] + 10:
            last["pct"] = pct
            log(f"  downloading: {pct}%")

    height = max_height or settings.max_source_height
    options = {
        "format": (
            f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]/"
            f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "concurrent_fragment_downloads": 4,
        "retries": 5,
        "progress_hooks": [hook],
    }
    log(f"  downloading {value} (max {height}p)")
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(value, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        candidates = sorted(workdir.glob("source.*"))
        if not candidates:
            raise RuntimeError("yt-dlp finished but no output file was produced")
        path = candidates[0]
    if path.suffix.lower() != ".mp4":
        converted = workdir / "source.mp4"
        run([FFMPEG, "-y", "-i", path, "-c", "copy", converted])
        path = converted
    log(f"  downloaded: {path.name} ({probe(path)['duration'] / 60:.1f} min)")
    return path


def extract_audio_chunk(source: Path, out: Path, start: float, duration: float) -> Path:
    """Extract a mono 16 kHz / 32 kbps mp3 chunk small enough for Groq's upload cap."""
    run(
        [
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{max(0.1, duration):.3f}",
            "-i", source,
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "libmp3lame", "-b:a", "32k",
            out,
        ],
        capture=True,
    )
    return out
