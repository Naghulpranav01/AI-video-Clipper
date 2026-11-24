from __future__ import annotations

import bisect
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings
from .providers import LLMConfig, default_llm_config, make_llm
from .util import clock, log, write_json


@dataclass
class Moment:
    start: float
    end: float
    title: str
    hook: str
    score: int
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


CONTENT_PRESETS: dict[str, dict] = {
    "general": {
        "name": "General viral",
        "description": "Balanced mix of the strongest moments of any kind.",
        "focus": "Pick the strongest moments of any kind: stories, hot takes, advice, emotion and humour.",
    },
    "podcast": {
        "name": "Podcast",
        "description": "Hot takes, disagreements, stories and quotable lines from hosts and guests.",
        "focus": "Favour bold opinions, disagreements, personal stories and quotable one-liners. Skip intros, ads, housekeeping and inside jokes that need the full episode.",
    },
    "interview": {
        "name": "Interview",
        "description": "Self-contained guest answers with emotion, surprise or revelation.",
        "focus": "Favour guest answers that stand alone, especially surprising admissions, turning points and emotional beats. Skip the interviewer's setup questions.",
    },
    "tutorial": {
        "name": "Tutorial / how-to",
        "description": "Complete actionable steps, tips and mistakes with a payoff.",
        "focus": "Favour complete actionable steps, concrete tips, common mistakes and before/after results. Each moment must teach one clear thing from setup to payoff.",
    },
    "comedy": {
        "name": "Comedy",
        "description": "Punchlines, reactions and absurd stories with a short setup.",
        "focus": "Favour punchlines, absurd stories, reactions and funny escalations. Keep the setup as short as possible and always keep the laugh in the clip.",
    },
    "business": {
        "name": "Business / finance",
        "description": "Contrarian business advice, numbers and lessons with proof.",
        "focus": "Favour contrarian business advice, surprising numbers, money stories and lessons backed by a concrete result or example. Skip generic motivation.",
    },
    "motivation": {
        "name": "Motivation",
        "description": "Mindset lines and transformation arcs with a tough-love edge.",
        "focus": "Favour transformative mindset statements, tough-love lessons and personal turnaround stories with a clear emotional arc.",
    },
}

DEFAULT_PRESET = "general"


def preset_catalog() -> list[dict]:
    return [
        {"key": key, "name": value["name"], "description": value["description"]}
        for key, value in CONTENT_PRESETS.items()
    ]


SYSTEM_PROMPT = """You are a senior short-form video editor who has cut thousands of viral clips from podcasts, interviews, streams and lectures.

You receive a timestamped transcript window from a long video. Find the strongest self-contained moments for TikTok, Reels and YouTube Shorts.

Editorial focus: {focus}

What makes a moment:
- It opens on something that grabs attention within the first 3 seconds: a bold claim, a story beat, tension, a surprising number, a mistake, a confession, a concrete tip.
- It is a complete thought. Start where the speaker starts the idea, end after the payoff. Never cut mid-sentence.
- It makes sense with zero context from earlier in the video.
- Prefer emotion, conflict, contrarian opinions, actionable advice, funny moments and story peaks. Skip intros, ads, rambling and "um"s.

Rules:
- start and end are absolute timestamps in seconds (floats), taken from the bracket numbers in the transcript.
- Each moment must last between {min_len} and {max_len} seconds.
- Moments must not overlap each other.
- score is 0-100, your predicted performance for short-form.
- title: under 80 characters, plain, specific, for YouTube.
- hook: under 55 characters, the on-screen opening line. Sentence case, no quotes, no hashtags, no emoji, no clickbait the clip does not deliver.
- reason: one short sentence on why it works.

Return JSON only, exactly this shape:
{{"moments": [{{"start": 742.5, "end": 785.0, "title": "The mistake everyone makes with pricing", "hook": "You are pricing for the wrong person", "score": 87, "reason": "Clear contrarian claim with a concrete fix."}}]}}"""


def build_blocks(segments: list[dict], max_seconds: float = 22.0, max_chars: int = 260) -> list[dict]:
    """Merge ASR segments into sentence-ish blocks that the LLM can cite by timestamp."""
    blocks: list[dict] = []
    current: dict | None = None
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if current is None:
            current = {"start": start, "end": end, "text": text}
            continue
        long_enough = (end - current["start"]) >= max_seconds or len(current["text"]) >= max_chars
        sentence_end = bool(re.search(r"[.!?][\"')\]]?$", current["text"]))
        if sentence_end or long_enough:
            blocks.append(current)
            current = {"start": start, "end": end, "text": text}
        else:
            current["text"] += " " + text
            current["end"] = end
    if current:
        blocks.append(current)
    return blocks


def build_windows(blocks: list[dict], char_budget: int) -> list[list[dict]]:
    windows: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for block in blocks:
        cost = len(block["text"]) + 16
        if current and size + cost > char_budget:
            windows.append(current)
            current, size = [], 0
        current.append(block)
        size += cost
    if current:
        windows.append(current)
    return windows


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


def _clean(text: str, limit: int) -> str:
    text = " ".join(str(text).replace("\n", " ").strip().strip("\"'").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")].rstrip(" ,.;:-") if " " in cut else cut


def _snap_to_segments(moment: Moment, segments: list[dict], tolerance: float = 2.5) -> Moment:
    if not segments:
        return moment
    starts = [float(s["start"]) for s in segments]
    i = bisect.bisect_right(starts, moment.start) - 1
    if i >= 0 and abs(starts[i] - moment.start) <= tolerance:
        moment.start = starts[i]
    j = bisect.bisect_right(starts, moment.end) - 1
    if j >= 0 and abs(float(segments[j]["end"]) - moment.end) <= tolerance:
        moment.end = float(segments[j]["end"])
    return moment


def _overlap(a: Moment, b: Moment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _select(candidates: list[Moment], count: int) -> list[Moment]:
    accepted: list[Moment] = []
    for moment in sorted(candidates, key=lambda m: m.score, reverse=True):
        if any(_overlap(moment, other) > max(3.0, 0.3 * moment.duration) for other in accepted):
            continue
        accepted.append(moment)
        if len(accepted) >= count:
            break
    return sorted(accepted, key=lambda m: m.start)


def _request_window(llm, window: list[dict], index: int, total: int, min_len: int, max_len: int, preset: str) -> list[Moment]:
    lines = [f"[{block['start']:.1f}] {block['text']}" for block in window]
    window_start, window_end = window[0]["start"], window[-1]["end"]
    ask = max(2, min(5, round((window_end - window_start) / 600) or 2))
    focus = CONTENT_PRESETS.get(preset, CONTENT_PRESETS[DEFAULT_PRESET])["focus"]
    system = SYSTEM_PROMPT.format(min_len=min_len, max_len=max_len, focus=focus)
    user = (
        f"Transcript window {index} of {total} (source time {clock(window_start)} to {clock(window_end)}).\n"
        f"Return up to {ask} moments. Every start/end must be between {window_start:.1f} and {window_end:.1f} seconds.\n\n"
        + "\n".join(lines)
    )

    log(f"  analyzing transcript window {index}/{total} (~{len(user) // 4} tokens)")
    content = llm.chat(system, user)
    try:
        data = _parse_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        log(f"  window {index}: unparseable model output ({exc})")
        return []

    moments: list[Moment] = []
    for raw in data.get("moments", []):
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(window_start - 2.0, start)
        end = min(window_end + 2.0, end)
        if end <= start:
            continue
        if end - start > max_len:
            end = start + max_len
        if end - start < min_len:
            end = min(end + (min_len - (end - start)), window_end + 2.0)
        if end - start < min_len * 0.8:
            continue
        try:
            score = int(round(float(raw.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        moments.append(
            Moment(
                start=start,
                end=end,
                title=_clean(raw.get("title", ""), 90) or "Untitled moment",
                hook=_clean(raw.get("hook", ""), 70),
                score=max(0, min(100, score)),
                reason=_clean(raw.get("reason", ""), 160),
            )
        )
    time.sleep(1.0 if index < total else 0.0)
    return moments


def pick_moments(
    transcript: dict,
    count: int,
    min_len: int = 15,
    max_len: int = 60,
    *,
    llm_config: LLMConfig | None = None,
    preset: str = DEFAULT_PRESET,
) -> list[Moment]:
    config = llm_config or default_llm_config()
    llm = make_llm(config)
    segments = transcript.get("segments") or []
    if not segments:
        raise RuntimeError("transcript has no segments")
    blocks = build_blocks(segments)
    windows = build_windows(blocks, settings.llm_char_budget)
    log(
        f"  {len(blocks)} transcript blocks in {len(windows)} window(s), "
        f"preset {preset}, model {config.resolved_provider()}/{config.resolved_model()}"
    )

    candidates: list[Moment] = []
    for i, window in enumerate(windows, start=1):
        candidates.extend(_request_window(llm, window, i, len(windows), min_len, max_len, preset))

    candidates = [_snap_to_segments(m, segments) for m in candidates]
    selected = _select(candidates, count)
    log(f"  {len(candidates)} candidate moments -> selected {len(selected)}")
    return selected


def save_moments(workdir: Path, moments: list[Moment]) -> None:
    write_json(workdir / "moments.json", {"moments": [asdict(m) for m in moments]})


def load_moments(path: Path) -> list[Moment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("moments", data) if isinstance(data, dict) else data
    return [
        Moment(
            start=float(item["start"]),
            end=float(item["end"]),
            title=str(item.get("title", "Untitled moment")),
            hook=str(item.get("hook", "")),
            score=int(item.get("score", 0) or 0),
            reason=str(item.get("reason", "")),
        )
        for item in raw
    ]
