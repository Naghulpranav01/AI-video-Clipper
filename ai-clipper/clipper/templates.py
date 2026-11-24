"""Caption / hook visual templates.

Assets use libass colour notation ``&HAABBGGRR``. PlayRes is always 1080x1920 so
the same ASS file scales cleanly to 720x1280 or 1080x1920 output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

WHITE = "&H00FFFFFF"
YELLOW = "&H0000FFFF"
BLACK = "&H00000000"
GREEN = "&H0000D74A"
RED = "&H000000D7"
DARK_RED = "&H000000B4"


@dataclass(frozen=True)
class Template:
    key: str
    name: str
    description: str
    font: str = "DejaVu Sans"
    size: int = 82
    uppercase: bool = True
    primary: str = WHITE
    highlight: str = YELLOW
    outline: str = BLACK
    outline_w: int = 8
    shadow: int = 0
    margin_v: int = 470
    max_words: int = 4
    max_chars: int = 20
    boxed: bool = False
    back: str = "&H64000000"
    hook_mode: str = "banner"  # banner | outline | none
    hook_size: int = 76
    hook_primary: str = BLACK
    hook_back: str = YELLOW

    def css(self) -> dict:
        return {
            "primary": ass_to_css(self.primary),
            "highlight": ass_to_css(self.highlight),
            "outline": ass_to_css(self.back if self.boxed else self.outline),
            "hook_fg": ass_to_css(self.hook_primary),
            "hook_bg": ass_to_css(self.hook_back),
            "boxed": self.boxed,
            "hook_mode": self.hook_mode,
            "font": self.font,
        }

    def public(self) -> dict:
        data = asdict(self)
        data.pop("primary", None)
        data.pop("highlight", None)
        data.pop("outline", None)
        data.pop("back", None)
        data.pop("hook_primary", None)
        data.pop("hook_back", None)
        data["style"] = self.css()
        return data


TEMPLATES: dict[str, Template] = {
    "bold-pop": Template(
        key="bold-pop",
        name="Bold Pop",
        description="White caps with a yellow karaoke word and yellow hook banner. The default all-rounder.",
    ),
    "clean-white": Template(
        key="clean-white",
        name="Clean White",
        description="Sentence-case white captions, no colour pop, minimal outlined hook. Calm and premium.",
        size=78,
        uppercase=False,
        highlight=WHITE,
        outline_w=7,
        margin_v=450,
        max_words=5,
        max_chars=26,
        hook_mode="outline",
        hook_size=64,
        hook_primary=WHITE,
        hook_back=BLACK,
    ),
    "karaoke-green": Template(
        key="karaoke-green",
        name="Karaoke Green",
        description="Tight three-word phrases with a green highlight word and yellow hook banner.",
        size=84,
        highlight=GREEN,
        outline_w=10,
        margin_v=470,
        max_words=3,
        max_chars=16,
        hook_size=78,
    ),
    "beast": Template(
        key="beast",
        name="Beast Mode",
        description="Huge yellow text, thick black outline, red hook banner. Maximum energy for shorts.",
        size=104,
        primary=YELLOW,
        highlight=WHITE,
        outline_w=15,
        margin_v=510,
        max_words=3,
        max_chars=15,
        hook_size=90,
        hook_primary=WHITE,
        hook_back=RED,
    ),
    "minimal": Template(
        key="minimal",
        name="Minimal",
        description="Small lowercase captions low on the frame with a light outlined hook. Documentary feel.",
        size=62,
        uppercase=False,
        highlight=WHITE,
        outline_w=6,
        margin_v=400,
        max_words=6,
        max_chars=34,
        hook_mode="outline",
        hook_size=58,
        hook_primary=WHITE,
        hook_back=BLACK,
    ),
    "news-box": Template(
        key="news-box",
        name="News Box",
        description="Boxed red caption bars with white text and a red headline banner. Broadcast look.",
        size=74,
        boxed=True,
        back=DARK_RED,
        outline_w=18,
        margin_v=440,
        max_words=4,
        max_chars=24,
        hook_size=70,
        hook_primary=WHITE,
        hook_back=DARK_RED,
    ),
}

DEFAULT_TEMPLATE = "bold-pop"


def get_template(key: str | None) -> Template:
    return TEMPLATES.get(str(key or DEFAULT_TEMPLATE), TEMPLATES[DEFAULT_TEMPLATE])


def ass_to_css(value: str) -> str:
    """Convert ``&HAABBGGRR`` (alpha ignored) to ``#RRGGBB`` for UI previews."""
    text = str(value).strip().lstrip("&").lstrip("hH")
    text = text[-8:].rjust(8, "0") if len(text) > 8 else text.rjust(8, "0")
    try:
        number = int(text, 16)
    except ValueError:
        return "#ffffff"
    b = (number >> 16) & 0xFF
    g = (number >> 8) & 0xFF
    r = number & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def template_catalog() -> list[dict]:
    return [tpl.public() for tpl in TEMPLATES.values()]
