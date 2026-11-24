from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import settings
from .moments import CONTENT_PRESETS
from .pipeline import ClipOptions, clip_video, rerender
from .providers import ASR_PROVIDERS, LLM_PROVIDERS, default_asr_config, default_llm_config
from .publish import publish_clip
from .render import QUALITIES
from .templates import TEMPLATES

LLM_CHOICES = [p["key"] for p in LLM_PROVIDERS]
ASR_CHOICES = [p["key"] for p in ASR_PROVIDERS]


def _add_render_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crop", choices=["track", "blur", "center"], default="track",
                        help="track: face-tracked 9:16 crop, blur: blurred background, center: fixed center crop")
    parser.add_argument("--template", choices=list(TEMPLATES), default=None,
                        help="caption/hook style (default: TEMPLATE from .env or bold-pop)")
    parser.add_argument("--quality", choices=list(QUALITIES), default=None,
                        help="turbo: 720p ultrafast, standard: 1080p (default: RENDER_QUALITY or standard)")
    parser.add_argument("--jobs", type=int, default=0, metavar="N",
                        help="clips to render in parallel (default: RENDER_CONCURRENCY or 2)")
    parser.add_argument("--no-captions", action="store_true", help="burn no subtitles")
    parser.add_argument("--no-hook", action="store_true", help="skip the opening hook overlay")


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-provider", choices=LLM_CHOICES, default=None,
                        help="groq | openai (any OpenAI-compatible URL) | gemini | ollama")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL (e.g. http://127.0.0.1:1234/v1)")
    parser.add_argument("--asr-provider", choices=ASR_CHOICES, default=None,
                        help="groq | openai | local (faster-whisper)")
    parser.add_argument("--asr-model", default=None, help="ASR model name (e.g. whisper-large-v3-turbo, base)")
    parser.add_argument("--asr-base-url", default=None, help="OpenAI-compatible base URL for ASR")


def _llm_config(args):
    config = default_llm_config()
    provider = getattr(args, "llm_provider", None)
    if provider and provider != config.provider:
        config.provider = provider
        config.model = ""
        config.api_key = ""
        config.base_url = ""
    else:
        config.provider = provider or config.provider
    if getattr(args, "llm_model", None):
        config.model = args.llm_model
    if getattr(args, "llm_base_url", None):
        config.base_url = args.llm_base_url
    return config


def _asr_config(args):
    config = default_asr_config()
    provider = getattr(args, "asr_provider", None)
    if provider and provider != config.provider:
        config.provider = provider
        config.model = ""
        config.api_key = ""
        config.base_url = ""
    else:
        config.provider = provider or config.provider
    if getattr(args, "asr_model", None):
        config.model = args.asr_model
    if getattr(args, "asr_base_url", None):
        config.base_url = args.asr_base_url
    if not config.language:
        config.language = settings.language
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Turn long videos into captioned 9:16 clips with Groq Whisper + Llama.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    clip = sub.add_parser("clip", help="download/transcribe a video and render clips")
    clip.add_argument("input", help="YouTube/URL or a local video file")
    clip.add_argument("-n", "--count", type=int, default=5, help="number of clips (default 5)")
    clip.add_argument("--min", dest="min_len", type=float, default=15.0, help="minimum clip seconds (default 15)")
    clip.add_argument("--max", dest="max_len", type=float, default=60.0, help="maximum clip seconds (default 60)")
    clip.add_argument("--out", type=Path, default=None, help="output folder (default ./output)")
    clip.add_argument("--preset", choices=list(CONTENT_PRESETS), default="general",
                      help="what kind of moments to look for (default: general)")
    clip.add_argument("--moments", type=Path, default=None, help="JSON file of moments to render instead of calling the LLM")
    clip.add_argument("--transcript", type=Path, default=None, help="reuse a transcript.json instead of transcribing")
    clip.add_argument("--force-transcribe", action="store_true", help="ignore cached transcript.json in the job folder")
    _add_render_flags(clip)
    _add_model_flags(clip)

    again = sub.add_parser("rerender", help="re-render an existing job with new framing/caption settings")
    again.add_argument("workdir", type=Path, help="job folder (contains source.mp4, transcript.json, moments.json)")
    again.add_argument("--moments", type=Path, default=None, help="override moments.json")
    _add_render_flags(again)

    only = sub.add_parser("transcribe", help="transcribe only and save transcript.json")
    only.add_argument("input", help="YouTube/URL or a local video file")
    only.add_argument("--out", type=Path, default=None, help="output folder (default ./output)")
    only.add_argument("--asr-provider", choices=ASR_CHOICES, default=None)
    only.add_argument("--asr-model", default=None)
    only.add_argument("--asr-base-url", default=None)

    serve = sub.add_parser("serve", help="run the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    post = sub.add_parser("publish", help="post a clip through Upload-Post")
    post.add_argument("video", type=Path)
    post.add_argument("--platforms", default="tiktok,instagram,youtube", help="comma separated")
    post.add_argument("--title", required=True)
    post.add_argument("--description", default="")
    post.add_argument("--user", default=None, help="Upload-Post profile name (default UPLOAD_POST_USER)")
    post.add_argument("--scheduled-date", default=None, help="ISO-8601 time to schedule the post")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "clip":
        options = ClipOptions(
            count=max(1, args.count),
            min_len=args.min_len,
            max_len=args.max_len,
            crop=args.crop,
            captions=not args.no_captions,
            hook=not args.no_hook,
            template=args.template,
            quality=args.quality,
            preset=args.preset,
            render_concurrency=args.jobs,
            transcript_file=str(args.transcript) if args.transcript else None,
            moments_file=str(args.moments) if args.moments else None,
            force_transcribe=args.force_transcribe,
            output_dir=args.out or settings.output_dir,
            llm_config=_llm_config(args),
            asr_config=_asr_config(args),
        )
        job = clip_video(args.input, options)
        _print_summary(job)
        return 0

    if args.command == "rerender":
        options = ClipOptions(
            crop=args.crop,
            captions=not args.no_captions,
            hook=not args.no_hook,
            template=args.template,
            quality=args.quality,
            render_concurrency=args.jobs,
            moments_file=str(args.moments) if args.moments else None,
        )
        job = rerender(args.workdir, options)
        _print_summary(job)
        return 0

    if args.command == "transcribe":
        from .ingest import fetch_source
        from .transcribe import transcribe
        from .util import log, probe

        out_root = Path(args.out or settings.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        workdir = out_root / "transcripts"
        workdir.mkdir(exist_ok=True)
        source = fetch_source(args.input, workdir, max_height=settings.max_source_height)
        info = probe(source)
        log(f"source: {info['duration'] / 60:.1f} min, {info['width']}x{info['height']}")
        transcript = transcribe(source, workdir, force=True, asr_config=_asr_config(args))
        log(f"wrote {workdir / 'transcript.json'} ({len(transcript['words'])} words)")
        return 0

    if args.command == "serve":
        from .web import run_server

        run_server(args.host, args.port)
        return 0

    if args.command == "publish":
        publish_clip(
            args.video,
            [p.strip() for p in args.platforms.split(",") if p.strip()],
            title=args.title,
            description=args.description,
            user=args.user,
            scheduled_date=args.scheduled_date,
        )
        return 0

    return 1


def _print_summary(job: dict) -> None:
    print()
    print(f"{len(job['clips'])} clips ready in {job['workdir']}")
    print(f"gallery: {Path(job['workdir']) / 'gallery.html'}")
    for clip in job["clips"]:
        print(f"  [{clip['score']:>3}] {clip['title']}  ({clip['duration']:.0f}s)  ->  {clip['file']}")


if __name__ == "__main__":
    sys.exit(main())
