from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _output_dir() -> Path:
    path = Path(os.getenv("OUTPUT_DIR", "output")).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).strip().rstrip("/")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").strip().lower())
    asr_provider: str = field(default_factory=lambda: os.getenv("ASR_PROVIDER", "groq").strip().lower())
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"))
    openai_llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    gemini_llm_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    ollama_llm_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    asr_model: str = field(default_factory=lambda: os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo"))
    openai_asr_model: str = field(default_factory=lambda: os.getenv("OPENAI_ASR_MODEL", "whisper-1"))
    local_whisper_model: str = field(default_factory=lambda: os.getenv("LOCAL_WHISPER_MODEL", "base"))
    language: str = field(default_factory=lambda: os.getenv("ASR_LANGUAGE", "").strip() or None)
    asr_chunk_seconds: int = field(default_factory=lambda: _env_int("ASR_CHUNK_SECONDS", 600))
    asr_concurrency: int = field(default_factory=lambda: _env_int("ASR_CONCURRENCY", 3))
    llm_char_budget: int = field(default_factory=lambda: _env_int("LLM_CHAR_BUDGET", 18000))
    output_dir: Path = field(default_factory=_output_dir)
    max_source_height: int = field(default_factory=lambda: _env_int("MAX_SOURCE_HEIGHT", 1080))
    render_template: str = field(default_factory=lambda: os.getenv("TEMPLATE", "bold-pop"))
    render_quality: str = field(default_factory=lambda: os.getenv("RENDER_QUALITY", "standard"))
    render_concurrency: int = field(default_factory=lambda: _env_int("RENDER_CONCURRENCY", 2))
    upload_post_key: str = field(default_factory=lambda: os.getenv("UPLOAD_POST_API_KEY", "").strip())
    upload_post_user: str = field(default_factory=lambda: os.getenv("UPLOAD_POST_USER", "").strip())
    allow_origins: str = field(default_factory=lambda: os.getenv("ALLOW_ORIGINS", "").strip())
    public_url: str = field(default_factory=lambda: os.getenv("PUBLIC_URL", "").strip().rstrip("/"))


settings = Settings()
