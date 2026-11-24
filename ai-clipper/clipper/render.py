from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .faces import face_track, keyframe_crop_x
from .moments import Moment
from .subtitles import write_subtitles
from .templates import DEFAULT_TEMPLATE, get_template
from .util import FFMPEG, clock, log, probe, run, run_ffmpeg, write_json


@dataclass(frozen=True)
class Quality:
    key: str
    name: str
    width: int
    height: int
    fps: int
    preset: str
    crf: int
    max_source_height: int


QUALITIES: dict[str, Quality] = {
    "turbo": Quality("turbo", "Turbo 720p", 720, 1280, 30, "ultrafast", 23, 720),
    "standard": Quality("standard", "Standard 1080p", 1080, 1920, 30, "veryfast", 20, 1080),
}
DEFAULT_QUALITY = "standard"


def get_quality(key: str | None) -> Quality:
    return QUALITIES.get(str(key or DEFAULT_QUALITY), QUALITIES[DEFAULT_QUALITY])


def quality_catalog() -> list[dict]:
    return [
        {
            "key": q.key,
            "name": q.name,
            "width": q.width,
            "height": q.height,
            "fps": q.fps,
            "preset": q.preset,
            "note": "fastest, smallest files" if q.key == "turbo" else "best quality, slower",
        }
        for q in QUALITIES.values()
    ]


@dataclass
class RenderOptions:
    crop: str = "track"  # track | blur | center
    captions: bool = True
    hook: bool = True
    template: str = DEFAULT_TEMPLATE
    quality: str = DEFAULT_QUALITY
    threads: int = 0


def _video_chain(src_w: int, src_h: int, x_expr: str, subtitles: bool, q: Quality) -> str:
    ratio = src_w / src_h if src_h else 16 / 9
    target_ratio = q.width / q.height
    if ratio >= target_ratio:
        chain = f"scale=-2:{q.height}:flags=lanczos,crop={q.width}:{q.height}:x='{x_expr}':y=0"
    else:
        scaled_h = int(round(src_h * q.width / src_w / 2) * 2)
        if scaled_h >= q.height:
            chain = f"scale={q.width}:-2:flags=lanczos,crop={q.width}:{q.height}:x=0:y='(ih-{q.height})/2'"
        else:
            chain = f"scale={q.width}:-2:flags=lanczos,pad={q.width}:{q.height}:0:'(oh-ih)/2':black"
    chain += f",setsar=1,fps={q.fps}"
    if subtitles:
        chain += ",ass=filename=captions.ass"
    return chain + "[v]"


def _blur_chain(subtitles: bool, q: Quality) -> str:
    tail = f",setsar=1,fps={q.fps}"
    if subtitles:
        tail += ",ass=filename=captions.ass"
    small_w, small_h = max(16, q.width // 4), max(16, q.height // 4)
    return (
        f"[0:v]scale={q.width}:{q.height}:force_original_aspect_ratio=increase,"
        f"crop={q.width}:{q.height},scale={small_w}:{small_h},boxblur=8:2,scale={q.width}:{q.height}[bg];"
        f"[0:v]scale={q.width}:-2:flags=lanczos[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto{tail}[v]"
    )


def _scaled_width(src_w: int, src_h: int, target_h: int) -> int:
    return int(round(src_w * target_h / src_h / 2) * 2)


def render_clip(
    source: Path,
    info: dict,
    moment: Moment,
    words: list[dict],
    output_root: Path,
    index: int,
    options: RenderOptions,
    progress=log,
) -> dict:
    q = get_quality(options.quality)
    tpl = get_template(options.template)
    clip_len = max(1.0, moment.end - moment.start)
    clip_dir = output_root / f"clip_{index:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.mp4"
    started = time.time()
    progress(
        f"[{index}] rendering {clock(moment.start)}-{clock(moment.end)} ({clip_len:.0f}s) "
        f"{moment.title!r} [{q.key}/{tpl.key}]"
    )

    hook_text = moment.hook if options.hook else ""
    has_subs = bool(hook_text.strip()) or (options.captions and bool(words))
    if options.captions and not words:
        progress("  no words in this range, rendering without captions")
    ass_path, srt_path = write_subtitles(
        clip_dir, words, moment.start, moment.end, hook_text, options.captions, template=tpl
    )

    subtitles_filter = has_subs and (options.captions or hook_text.strip())
    x_expr = "0"
    src_w, src_h = info["width"], info["height"]
    if options.crop == "track" and src_w and src_h and src_w / src_h >= q.width / q.height:
        samples = face_track(source, moment.start, clip_len, clip_dir / "track")
        if samples:
            scaled_w = _scaled_width(src_w, src_h, q.height)
            x_expr = keyframe_crop_x(samples, clip_len, scaled_w, q.width)
            progress(f"  crop pan built from {len(samples)} samples")

    if options.crop == "blur":
        video = _blur_chain(subtitles_filter, q)
    else:
        video = f"[0:v]{_video_chain(src_w, src_h, x_expr, subtitles_filter, q)}"

    graph = video
    maps = ["-map", "[v]"]
    if info.get("has_audio"):
        # dynaudnorm is ~20% faster than loudnorm per clip and keeps speech even
        graph += ";[0:a]dynaudnorm=f=200:g=15:p=0.95[a]"
        maps += ["-map", "[a]"]

    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-ss", f"{moment.start:.3f}",
        "-t", f"{clip_len:.3f}",
        "-i", source.resolve(),
        "-filter_complex", graph,
        *maps,
        "-c:v", "libx264", "-preset", q.preset, "-crf", str(q.crf),
        "-pix_fmt", "yuv420p", "-r", str(q.fps),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
    ]
    if options.threads > 0:
        cmd += ["-threads", str(options.threads)]
    cmd += ["-progress", "pipe:1", "clip.mp4"]
    run_ffmpeg(cmd, total_seconds=clip_len, label=f"clip {index}", cwd=clip_dir)

    thumb = clip_dir / "thumbnail.jpg"
    thumb_at = min(1.2, max(0.2, clip_len / 2))
    run(
        [FFMPEG, "-y", "-v", "error", "-ss", f"{thumb_at:.2f}", "-i", clip_path, "-frames:v", "1", "-q:v", "3", thumb],
        capture=True,
    )

    rendered = probe(clip_path)
    meta = {
        "index": index,
        "title": moment.title,
        "hook": moment.hook,
        "score": moment.score,
        "reason": moment.reason,
        "start": round(moment.start, 2),
        "end": round(moment.end, 2),
        "duration": round(rendered["duration"], 2),
        "width": rendered["width"],
        "height": rendered["height"],
        "file": f"clip_{index:02d}/clip.mp4",
        "thumbnail": f"clip_{index:02d}/thumbnail.jpg",
        "captions": f"clip_{index:02d}/captions.srt",
        "ass": str(ass_path.relative_to(output_root)),
        "crop": options.crop,
        "template": tpl.key,
        "quality": q.key,
    }
    write_json(clip_dir / "meta.json", meta)
    progress(
        f"  wrote {clip_path.relative_to(output_root)} "
        f"({rendered['duration']:.1f}s, {rendered['width']}x{rendered['height']}, {time.time() - started:.0f}s render)"
    )
    return meta
