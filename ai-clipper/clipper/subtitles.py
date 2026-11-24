from __future__ import annotations

import os
from pathlib import Path

from .templates import BLACK, Template, get_template
from .util import ass_time, srt_time

HOOK_SECONDS = float(os.getenv("HOOK_SECONDS", "3.6"))


def _escape(text: str) -> str:
    return str(text).replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _clip_words(words: list[dict], start: float, end: float) -> list[dict]:
    out = []
    for word in words:
        mid = (float(word["start"]) + float(word["end"])) / 2
        if start - 0.15 <= mid < end + 0.15:
            text = str(word["word"]).strip()
            if not text:
                continue
            out.append(
                {
                    "word": text,
                    "start": max(0.0, float(word["start"]) - start),
                    "end": max(0.0, float(word["end"]) - start),
                }
            )
    return out


def group_phrases(words: list[dict], max_words: int = 4, max_chars: int = 20, max_gap: float = 0.7) -> list[list[dict]]:
    phrases: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if current:
            text_len = sum(len(w["word"]) for w in current) + len(current)
            gap = word["start"] - current[-1]["end"]
            if (
                len(current) >= max_words
                or text_len + len(word["word"]) > max_chars
                or gap > max_gap
                or current[-1]["word"].endswith((".", "!", "?", ","))
            ):
                phrases.append(current)
                current = []
        current.append(word)
    if current:
        phrases.append(current)
    return phrases


def _hook_lines(hook: str, max_chars: int = 22, max_lines: int = 3) -> list[str]:
    words = hook.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    # avoid a one-word orphan line under a full-width line
    if len(lines) >= 2 and len(lines[-1]) <= 7 and len(lines[-2]) > 12 and len(lines[-2].split()) > 1:
        moved = lines[-2].rsplit(" ", 1)[1]
        lines[-2] = lines[-2].rsplit(" ", 1)[0]
        lines[-1] = f"{moved} {lines[-1]}"
    return lines[:max_lines]


def _ass_header(tpl: Template) -> str:
    spacing = 1 if tpl.uppercase else 0
    cap_border = 3 if tpl.boxed else 1
    cap_outline_colour = tpl.back if tpl.boxed else tpl.outline
    cap_shadow = 0 if tpl.boxed else tpl.shadow
    if tpl.hook_mode == "banner":
        hook_style = (3, max(10, tpl.hook_size // 5), tpl.hook_back)
    elif tpl.hook_mode == "outline":
        hook_style = (1, 8, BLACK)
    else:
        hook_style = (1, 0, BLACK)
    hook_border, hook_outline, hook_outline_colour = hook_style
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{tpl.font},{tpl.size},{tpl.primary},{tpl.primary},{cap_outline_colour},&H64000000,-1,0,0,0,100,100,{spacing},0,{cap_border},{tpl.outline_w},{cap_shadow},2,70,70,{tpl.margin_v},1
Style: Hook,{tpl.font},{tpl.hook_size},{tpl.hook_primary},{tpl.hook_primary},{hook_outline_colour},&H00000000,-1,0,0,0,100,100,0,0,{hook_border},{hook_outline},0,8,80,80,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    hook: str,
    *,
    show_captions: bool = True,
    template: str | Template = "bold-pop",
) -> str:
    tpl = template if isinstance(template, Template) else get_template(template)
    clip_len = max(0.1, clip_end - clip_start)
    relative = _clip_words(words, clip_start, clip_end)
    events: list[str] = []

    if hook.strip() and tpl.hook_mode != "none":
        text = hook.strip().upper() if tpl.uppercase else hook.strip()
        max_chars = max(12, int(1000 / (tpl.hook_size * 0.58)))
        lines = _hook_lines(text, max_chars=max_chars)
        banner = "\\N".join(_escape(line) for line in lines)
        end = min(clip_len, HOOK_SECONDS)
        events.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Hook,,0,0,0,,"
            f"{{\\fad(180,260)\\fscx96\\fscy96\\t(0,180,\\fscx100\\fscy100)}}{banner}"
        )

    if show_captions:
        phrases = group_phrases(relative, max_words=tpl.max_words, max_chars=tpl.max_chars)
        prev_end = 0.0
        for i, phrase in enumerate(phrases):
            if not phrase:
                continue
            p_start = max(0.0, phrase[0]["start"] - 0.04, prev_end + 0.02 if i > 0 else 0.0)
            next_start = phrases[i + 1][0]["start"] - 0.03 if i + 1 < len(phrases) else clip_len
            p_end = min(max(phrase[-1]["end"], p_start + 0.2), next_start)
            p_end = max(p_end, p_start + 0.06)
            prev_end = p_end
            starts: list[float] = []
            for j, word in enumerate(phrase):
                start = p_start if j == 0 else max(p_start, word["start"] - 0.03, starts[-1] + 0.01)
                starts.append(min(start, p_end - 0.04))
            for j, word in enumerate(phrase):
                start = starts[j]
                end = starts[j + 1] if j + 1 < len(starts) else p_end
                end = max(end, start + 0.04)
                parts = []
                for k, item in enumerate(phrase):
                    token = _escape(item["word"].upper() if tpl.uppercase else item["word"])
                    colour = tpl.highlight if k == j else tpl.primary
                    parts.append(f"{{\\c{colour}}}{token}")
                prefix = "{\\fad(50,0)}" if j == 0 else ""
                events.append(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{prefix}{' '.join(parts)}"
                )

    return _ass_header(tpl) + "\n".join(events) + "\n"


def build_srt(words: list[dict], clip_start: float, clip_end: float, template: str | Template = "bold-pop") -> str:
    tpl = template if isinstance(template, Template) else get_template(template)
    relative = _clip_words(words, clip_start, clip_end)
    cues = []
    for phrase in group_phrases(relative, max_words=tpl.max_words, max_chars=tpl.max_chars):
        if not phrase:
            continue
        text = " ".join(_escape(w["word"]) for w in phrase)
        start = phrase[0]["start"]
        end = min(clip_end - clip_start, phrase[-1]["end"] + 0.1)
        cues.append(f"{len(cues) + 1}\n{srt_time(start)} --> {srt_time(max(end, start + 0.3))}\n{text}\n")
    return "\n".join(cues)


def write_subtitles(
    clip_dir: Path,
    words: list[dict],
    clip_start: float,
    clip_end: float,
    hook: str,
    captions: bool,
    template: str | Template = "bold-pop",
) -> tuple[Path, Path]:
    ass_path = clip_dir / "captions.ass"
    srt_path = clip_dir / "captions.srt"
    ass_path.write_text(build_ass(words, clip_start, clip_end, hook, show_captions=captions, template=template), encoding="utf-8")
    srt_path.write_text(build_srt(words, clip_start, clip_end, template=template), encoding="utf-8")
    return ass_path, srt_path
