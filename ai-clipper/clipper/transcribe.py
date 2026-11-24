from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import settings
from .ingest import extract_audio_chunk
from .providers import ASRConfig, default_asr_config, make_asr
from .util import dedupe_words, log, probe, write_json


def _transcribe_chunk(asr, audio: Path, offset: float, index: int, total: int) -> dict:
    log(f"  transcribing chunk {index}/{total} ({offset / 60:.0f} min in)")
    data = asr.transcribe(audio)
    words = data.get("words") or []
    segments = data.get("segments") or []
    for word in words:
        word["start"] = float(word["start"]) + offset
        word["end"] = float(word["end"]) + offset
    for segment in segments:
        segment["start"] = float(segment["start"]) + offset
        segment["end"] = float(segment["end"]) + offset
    return {"words": words, "segments": segments, "language": data.get("language")}


def transcribe(
    source: Path,
    workdir: Path,
    *,
    force: bool = False,
    asr_config: ASRConfig | None = None,
) -> dict:
    cache = workdir / "transcript.json"
    if cache.exists() and not force:
        log(f"  using cached transcript ({cache.name})")
        return json.loads(cache.read_text(encoding="utf-8"))

    config = asr_config or default_asr_config()
    asr = make_asr(config)
    duration = probe(source)["duration"]
    if duration <= 0:
        raise RuntimeError("could not read source duration")

    chunk = max(60, settings.asr_chunk_seconds)
    count = max(1, math.ceil(duration / chunk))
    workers = max(1, min(config.worker_count(), count))
    log(
        f"  audio: {duration / 60:.1f} min, {count} chunk(s) of {chunk // 60} min, "
        f"{config.resolved_provider()} / {config.resolved_model()} x{workers}"
    )

    chunks_dir = workdir / "audio"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for i in range(count):
        start = i * chunk
        length = min(chunk, duration - start)
        if length <= 0.5:
            continue
        plan.append((i, start, length, chunks_dir / f"chunk_{i:03d}.mp3"))

    for i, start, length, path in plan:
        if not path.exists() or path.stat().st_size == 0:
            extract_audio_chunk(source, path, start, length)

    results: dict[int, dict] = {}
    if workers == 1:
        for i, start, _length, path in plan:
            results[i] = _transcribe_chunk(asr, path, start, i + 1, len(plan))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_transcribe_chunk, asr, path, start, i + 1, len(plan)): i
                for i, start, _length, path in plan
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    words: list[dict] = []
    segments: list[dict] = []
    language = None
    for i in sorted(results):
        words.extend(results[i]["words"])
        segments.extend(results[i]["segments"])
        language = language or results[i].get("language")

    words = dedupe_words(words)
    segments.sort(key=lambda s: s["start"])
    payload = {"language": language, "duration": duration, "segments": segments, "words": words}
    write_json(cache, payload)
    log(f"  transcript: {len(words)} words, language {language or 'unknown'}")
    return payload
