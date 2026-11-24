"""Verify the face-tracked crop on a generated photo that pans across the frame.

    .venv/bin/python tests/smoke_tracking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clipper.faces import face_track, keyframe_crop_x  # noqa: E402
from clipper.moments import Moment  # noqa: E402
from clipper.render import RenderOptions, render_clip  # noqa: E402
from clipper.util import FFMPEG, probe, run  # noqa: E402

OUT = ROOT / "output" / "tracking"
IMAGE = ROOT / "assets" / "demo-wide-frame.png"


def make_pan_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG, "-y", "-v", "error",
            "-loop", "1", "-i", IMAGE,
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000",
            "-t", "20",
            "-vf", "scale=2560:-2,crop=1280:720:x='(iw-ow)*t/20':y=0",
            "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            path,
        ],
        capture=True,
    )


def main() -> int:
    if not IMAGE.exists():
        raise SystemExit(f"missing test image: {IMAGE}")
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "pan.mp4"
    if not source.exists():
        make_pan_video(source)
    info = probe(source)
    print(f"source {info['width']}x{info['height']} {info['duration']:.1f}s")

    clip_dir = OUT / "job"
    (clip_dir / "clip_01").mkdir(parents=True, exist_ok=True)
    samples = face_track(source, 0.0, 20.0, clip_dir / "track")
    print("track samples:", [(round(t, 1), round(x, 2)) for t, x in (samples or [])][::4])
    if samples:
        scaled_w = int(round(info["width"] * 1920 / info["height"] / 2) * 2)
        expr = keyframe_crop_x(samples, 20.0, scaled_w, 1080)
        print(f"crop expression: {len(expr)} chars, scaled_w={scaled_w}")
        assert "if(" in expr

    words = [
        {"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35}
        for i, w in enumerate("this is what a face tracked vertical crop looks like when the speaker moves across a wide frame".split())
    ]
    moment = Moment(0, 20, "Face tracked crop test", "Watch the frame follow me", 88)
    meta = render_clip(source, info, moment, words, clip_dir, 1, RenderOptions(crop="track"))
    print(meta)

    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    for at in (2, 9, 16):
        run(
            [FFMPEG, "-y", "-v", "error", "-ss", str(at), "-i", clip_dir / meta["file"],
             "-frames:v", "1", "-q:v", "2", frames / f"t{at:02d}.jpg"],
            capture=True,
        )
    print("frames in", frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
