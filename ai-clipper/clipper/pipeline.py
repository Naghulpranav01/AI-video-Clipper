from __future__ import annotations

import html
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import settings
from .ingest import fetch_source, is_url
from .moments import DEFAULT_PRESET, Moment, load_moments, pick_moments, save_moments
from .providers import ASRConfig, LLMConfig, default_asr_config, default_llm_config
from .render import RenderOptions, get_quality, render_clip
from .transcribe import transcribe
from .util import clock, log, probe, slugify, write_json


@dataclass
class ClipOptions:
    count: int = 5
    min_len: float = 15.0
    max_len: float = 60.0
    crop: str = "track"
    captions: bool = True
    hook: bool = True
    template: str | None = None
    quality: str | None = None
    preset: str = DEFAULT_PRESET
    render_concurrency: int = 0
    transcript_file: str | None = None
    moments_file: str | None = None
    force_transcribe: bool = False
    output_dir: Path | None = None
    llm_config: LLMConfig | None = None
    asr_config: ASRConfig | None = None


def _new_workdir(source: str, out_root: Path) -> Path:
    name = Path(source).stem if not is_url(source) else source.rstrip("/").split("/")[-1]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = out_root / f"{slugify(name, 40)}-{stamp}"
    workdir.mkdir(parents=True, exist_ok=False)
    return workdir


def _render_all(
    source: Path,
    info: dict,
    transcript: dict,
    moments: list[Moment],
    workdir: Path,
    options: ClipOptions,
    progress=log,
) -> list[dict]:
    words = transcript.get("words") or []
    quality = options.quality or settings.render_quality
    template = options.template or settings.render_template
    concurrency = max(1, options.render_concurrency or settings.render_concurrency)
    workers = max(1, min(concurrency, len(moments), os.cpu_count() or 1))
    threads = max(1, (os.cpu_count() or 4) // workers) if workers > 1 else 0
    render_options = RenderOptions(
        crop=options.crop,
        captions=options.captions,
        hook=options.hook,
        template=template,
        quality=quality,
        threads=threads,
    )
    progress(
        f"Rendering {len(moments)} clip(s) with {workers} parallel renderer(s), "
        f"{quality} quality, {template} template"
    )

    if workers == 1:
        return [
            render_clip(source, info, moment, words, workdir, index, render_options, progress)
            for index, moment in enumerate(moments, start=1)
        ]

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(render_clip, source, info, moment, words, workdir, index, render_options, progress): index
            for index, moment in enumerate(moments, start=1)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in sorted(results)]


def _job_payload(source_input: str, workdir: Path, info: dict, transcript: dict, moments, clips, options, elapsed: float) -> dict:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source_input,
        "source_file": str(workdir / "source.mp4"),
        "duration": round(info.get("duration", 0.0), 2),
        "width": info.get("width"),
        "height": info.get("height"),
        "language": transcript.get("language"),
        "word_count": len(transcript.get("words") or []),
        "options": {
            "crop": options.crop,
            "captions": options.captions,
            "hook": options.hook,
            "count": options.count,
            "min_len": options.min_len,
            "max_len": options.max_len,
            "template": options.template or settings.render_template,
            "quality": options.quality or settings.render_quality,
            "preset": options.preset,
            "render_concurrency": options.render_concurrency or settings.render_concurrency,
        },
        "models": {
            "llm": (options.llm_config or default_llm_config()).resolved_provider(),
            "llm_model": (options.llm_config or default_llm_config()).resolved_model(),
            "asr": (options.asr_config or default_asr_config()).resolved_provider(),
            "asr_model": (options.asr_config or default_asr_config()).resolved_model(),
        },
        "moments": [asdict(m) for m in moments],
        "clips": clips,
        "elapsed_seconds": round(elapsed, 1),
        "workdir": str(workdir),
    }


def clip_video(source_input: str, options: ClipOptions, progress=log) -> dict:
    started = time.time()
    llm_config = options.llm_config or default_llm_config()
    asr_config = options.asr_config or default_asr_config()
    out_root = Path(options.output_dir or settings.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    workdir = _new_workdir(source_input, out_root)
    progress(f"Job folder: {workdir}")

    quality = get_quality(options.quality or settings.render_quality)
    progress("Step 1/4  fetching source")
    source = fetch_source(source_input, workdir, max_height=min(settings.max_source_height, quality.max_source_height))
    local_copy = workdir / "source.mp4"
    if source.resolve() != local_copy.resolve():
        try:
            os.link(source, local_copy)
        except OSError:
            shutil.copy2(source, local_copy)
        source = local_copy
    info = probe(source)
    if info["duration"] <= 0 or not info["width"]:
        raise RuntimeError(f"could not read a video stream from {source}")
    progress(f"  {clock(info['duration'])} long, {info['width']}x{info['height']}")
    if not info.get("has_audio"):
        raise RuntimeError("source has no audio track; moment detection needs speech")

    progress(
        f"Step 2/4  transcribing with {asr_config.resolved_provider()}/{asr_config.resolved_model()}"
    )
    if options.transcript_file:
        transcript = json.loads(Path(options.transcript_file).read_text(encoding="utf-8"))
        write_json(workdir / "transcript.json", transcript)
        progress(f"  loaded transcript from {options.transcript_file}")
    else:
        transcript = transcribe(source, workdir, force=options.force_transcribe, asr_config=asr_config)

    progress(
        f"Step 3/4  finding viral moments with {llm_config.resolved_provider()}/{llm_config.resolved_model()} "
        f"({options.preset} preset)"
    )
    if options.moments_file:
        moments = load_moments(Path(options.moments_file))
        progress(f"  loaded {len(moments)} moments from {options.moments_file}")
    else:
        moments = pick_moments(
            transcript,
            options.count,
            options.min_len,
            options.max_len,
            llm_config=llm_config,
            preset=options.preset,
        )
    save_moments(workdir, moments)
    if not moments:
        raise RuntimeError("no moments were selected; try a longer video or a different model")

    progress("Step 4/4  reframing and rendering clips")
    clips = _render_all(source, info, transcript, moments, workdir, options, progress)

    job = _job_payload(source_input, workdir, info, transcript, moments, clips, options, time.time() - started)
    write_json(workdir / "job.json", job)
    write_gallery(workdir, job)
    progress(f"Done in {job['elapsed_seconds']}s: {len(clips)} clips in {workdir}")
    return job


def rerender(workdir: Path, options: ClipOptions, progress=log) -> dict:
    """Re-render an existing job with new framing/caption options (transcript is reused)."""
    started = time.time()
    workdir = Path(workdir).resolve()
    source = workdir / "source.mp4"
    if not source.exists():
        raise SystemExit(f"no source.mp4 in {workdir}")
    transcript_path = workdir / "transcript.json"
    if not transcript_path.exists():
        raise SystemExit(f"no transcript.json in {workdir}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    moments = load_moments(Path(options.moments_file) if options.moments_file else workdir / "moments.json")
    info = probe(source)
    previous = workdir / "job.json"
    if previous.exists():
        try:
            old = json.loads(previous.read_text(encoding="utf-8")).get("options", {})
            options.count = int(old.get("count", options.count))
            options.min_len = float(old.get("min_len", options.min_len))
            options.max_len = float(old.get("max_len", options.max_len))
            options.template = options.template or old.get("template")
            options.quality = options.quality or old.get("quality")
        except (ValueError, TypeError):
            pass
    progress(
        f"Re-rendering {len(moments)} clips in {workdir} "
        f"({options.crop}, {options.template or settings.render_template}, captions={options.captions})"
    )
    clips = _render_all(source, info, transcript, moments, workdir, options, progress)
    job = _job_payload(str(workdir), workdir, info, transcript, moments, clips, options, time.time() - started)
    write_json(workdir / "job.json", job)
    write_gallery(workdir, job)
    progress(f"Done in {job['elapsed_seconds']}s")
    return job


GALLERY_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 40px 24px 80px; background: #0b0d10; color: #e8ecf1;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans", sans-serif; }
header { max-width: 1200px; margin: 0 auto 32px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }
.sub { color: #8b95a5; font-size: 14px; }
.grid { max-width: 1200px; margin: 0 auto; display: grid; gap: 22px;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.card { background: #14181e; border: 1px solid #222a34; border-radius: 14px; overflow: hidden;
  display: flex; flex-direction: column; }
.video { background: #000; display: flex; justify-content: center; }
video { width: 100%; max-height: 480px; display: block; }
.body { padding: 14px 16px 18px; display: flex; flex-direction: column; gap: 8px; flex: 1; }
.title { font-weight: 600; font-size: 15px; line-height: 1.35; }
.hook { color: #ffd75c; font-size: 13px; }
.meta { display: flex; gap: 10px; align-items: center; font-size: 12px; color: #8b95a5; }
.score { background: #1f6f43; color: #d6ffe8; border-radius: 999px; padding: 2px 9px; font-weight: 600; }
.links { margin-top: auto; display: flex; gap: 14px; font-size: 13px; }
a { color: #6fb3ff; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def write_gallery(workdir: Path, job: dict) -> Path:
    cards = []
    for clip in job["clips"]:
        title = html.escape(clip["title"])
        hook = html.escape(clip["hook"])
        cards.append(
            f"""
      <article class="card">
        <div class="video">
          <video controls preload="metadata" poster="{html.escape(clip['thumbnail'])}">
            <source src="{html.escape(clip['file'])}" type="video/mp4">
          </video>
        </div>
        <div class="body">
          <div class="title">{title}</div>
          <div class="hook">{hook}</div>
          <div class="meta"><span class="score">{clip['score']}</span>
            <span>{clock(clip['start'])} - {clock(clip['end'])}</span>
            <span>{clip['duration']:.0f}s</span></div>
          <div class="links">
            <a href="{html.escape(clip['file'])}" download>Download mp4</a>
            <a href="{html.escape(clip['captions'])}" download>SRT</a>
          </div>
        </div>
      </article>"""
        )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clips - {html.escape(Path(job['source_file']).name)}</title>
<style>{GALLERY_CSS}</style></head>
<body>
<header>
  <h1>Your clips</h1>
  <div class="sub">{len(job['clips'])} clips from {html.escape(job['source'])} · source {clock(job['duration'])} ·
  rendered in {job['elapsed_seconds']:.0f}s · {html.escape(job['created'])}</div>
</header>
<main class="grid">{''.join(cards)}
</main>
</body></html>
"""
    path = workdir / "gallery.html"
    path.write_text(page, encoding="utf-8")
    return path
