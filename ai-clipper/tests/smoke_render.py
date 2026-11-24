"""Offline smoke test: renders a clip from a generated video with fake timings.

No Groq key needed. Verifies ffmpeg, libass, caption generation, crop geometry and
the moment/subtitle helpers end to end.

    .venv/bin/python tests/smoke_render.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clipper.moments import Moment, _select, build_blocks, build_windows  # noqa: E402
from clipper.render import RenderOptions, render_clip  # noqa: E402
from clipper.subtitles import build_ass, group_phrases  # noqa: E402
from clipper.templates import TEMPLATES  # noqa: E402
from clipper.util import FFMPEG, probe, run  # noqa: E402

OUT = ROOT / "output" / "smoke"
SRC_DIR = OUT / "src"
CLIP_DIR = OUT / "job"
QUALITY = os.getenv("SMOKE_QUALITY", "standard")
TEMPLATE = os.getenv("SMOKE_TEMPLATE", "bold-pop")


def make_source(path: Path, seconds: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG, "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
            "-t", str(seconds),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            path,
        ],
        capture=True,
    )


def fake_transcript(start: float, end: float) -> list[dict]:
    lines = (
        "most people price their product by looking at competitors and guessing "
        "but the real number comes from the value the buyer gets not what the market charges "
        "so ask what changes for them after they buy and price against that outcome "
        "that is how you stop competing on price and start competing on results"
    ).split()
    span = (end - start) / len(lines)
    return [
        {"word": word, "start": start + i * span, "end": start + (i + 1) * span}
        for i, word in enumerate(lines)
    ]


def check_helpers() -> None:
    blocks = build_blocks(
        [
            {"start": 0.0, "end": 2.0, "text": "Hello there."},
            {"start": 2.0, "end": 4.0, "text": "This is a test."},
            {"start": 4.0, "end": 9.0, "text": "And another sentence."},
        ]
    )
    assert blocks and blocks[0]["text"].startswith("Hello"), blocks
    assert build_windows(blocks, 50), "windows should not be empty"
    a = Moment(10, 30, "A", "", 70)
    b = Moment(12, 28, "B", "", 50)
    c = Moment(40, 70, "C", "", 90)
    picked = _select([a, b, c], 5)
    assert [m.title for m in picked] == ["A", "C"], [m.title for m in picked]
    words = fake_transcript(0, 6)
    assert group_phrases(words), "phrases should not be empty"
    ass = build_ass(words, 0, 6, "WORDS ARE CHEAP", show_captions=True)
    assert "[Events]" in ass and "Dialogue:" in ass and "Hook" in ass
    for key in TEMPLATES:
        styled = build_ass(words, 0, 6, "TEMPLATE HOOK", show_captions=True, template=key)
        assert "Style: Cap" in styled and "Dialogue:" in styled, key
    boxed = build_ass(words, 0, 6, "BOX", template="news-box")
    assert "&H000000B4" in boxed, "news-box should use the dark red box colour"
    print("helper checks ok")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    check_helpers()

    source = SRC_DIR / "demo.mp4"
    if not source.exists():
        print("generating demo video...")
        make_source(source)
    info = probe(source)
    assert info["width"] == 1280 and info["height"] == 720, info

    start, end = 5.0, 22.0
    words = fake_transcript(4.0, 24.0)
    moment = Moment(
        start=start,
        end=end,
        title="Price against the outcome, not the market",
        hook="You are pricing for the wrong person",
        score=91,
        reason="Contrarian claim with a concrete fix.",
    )

    for crop in ("center", "track", "blur"):
        mode_dir = OUT / f"job_{crop}"
        (mode_dir / "clip_01").mkdir(parents=True, exist_ok=True)
        meta = render_clip(
            source, info, moment, words, mode_dir, 1,
            RenderOptions(crop=crop, quality=QUALITY, template=TEMPLATE),
        )
        rendered = probe(mode_dir / meta["file"])
        expected = (720, 1280) if QUALITY == "turbo" else (1080, 1920)
        assert (rendered["width"], rendered["height"]) == expected, rendered
        assert 15.0 <= rendered["duration"] <= 18.0, rendered
        frames = mode_dir / "frames"
        frames.mkdir(exist_ok=True)
        run(
            [FFMPEG, "-y", "-v", "error", "-ss", "1.0", "-i", mode_dir / meta["file"],
             "-frames:v", "1", "-q:v", "2", frames / "start.jpg"],
            capture=True,
        )
        run(
            [FFMPEG, "-y", "-v", "error", "-ss", "8.0", "-i", mode_dir / meta["file"],
             "-frames:v", "1", "-q:v", "2", frames / "mid.jpg"],
            capture=True,
        )
        print(f"{crop}: {meta['duration']:.1f}s {meta['width']}x{meta['height']} ok")

    print(json.dumps({"ok": True, "frames": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
