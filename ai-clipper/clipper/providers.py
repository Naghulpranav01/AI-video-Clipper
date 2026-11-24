"""Pluggable LLM and ASR providers.

Online providers speak the OpenAI-compatible HTTP API (Groq, OpenAI, OpenRouter,
Together, LM Studio, vLLM, ...) plus native Gemini. Local LLMs go through
Ollama / LM Studio. Local ASR uses faster-whisper (CTranslate2, CPU).

API keys are only held in memory for the duration of a job. They are never
written to disk or logged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import settings
from .util import log

DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1:8b",
}

DEFAULT_ASR_MODELS = {
    "groq": "whisper-large-v3-turbo",
    "openai": "whisper-1",
    "local": "base",
}

LLM_PROVIDERS = [
    {
        "key": "groq",
        "name": "Groq",
        "kind": "online",
        "base_url": DEFAULT_BASE_URLS["groq"],
        "default_model": DEFAULT_MODELS["groq"],
        "needs_key": True,
        "hint": "Very fast and has a free tier. Needs a Groq API key.",
    },
    {
        "key": "openai",
        "name": "OpenAI-compatible",
        "kind": "online",
        "base_url": DEFAULT_BASE_URLS["openai"],
        "default_model": DEFAULT_MODELS["openai"],
        "needs_key": True,
        "hint": "OpenAI, OpenRouter, Together, DeepSeek, LM Studio, vLLM ... set the base URL.",
    },
    {
        "key": "gemini",
        "name": "Google Gemini",
        "kind": "online",
        "base_url": DEFAULT_BASE_URLS["gemini"],
        "default_model": DEFAULT_MODELS["gemini"],
        "needs_key": True,
        "hint": "Free tier available at aistudio.google.com.",
    },
    {
        "key": "ollama",
        "name": "Ollama (local)",
        "kind": "local",
        "base_url": DEFAULT_BASE_URLS["ollama"],
        "default_model": DEFAULT_MODELS["ollama"],
        "needs_key": False,
        "hint": "Runs fully offline. Start Ollama and pull a model first.",
    },
]

ASR_PROVIDERS = [
    {
        "key": "groq",
        "name": "Groq Whisper",
        "kind": "online",
        "base_url": DEFAULT_BASE_URLS["groq"],
        "default_model": DEFAULT_ASR_MODELS["groq"],
        "needs_key": True,
        "hint": "whisper-large-v3-turbo, fast and cheap. Same key as Groq LLM.",
    },
    {
        "key": "openai",
        "name": "OpenAI Whisper / compatible",
        "kind": "online",
        "base_url": DEFAULT_BASE_URLS["openai"],
        "default_model": DEFAULT_ASR_MODELS["openai"],
        "needs_key": True,
        "hint": "Any server exposing /audio/transcriptions (OpenAI, Groq, local whisper.cpp).",
    },
    {
        "key": "local",
        "name": "faster-whisper (local)",
        "kind": "local",
        "base_url": "",
        "default_model": DEFAULT_ASR_MODELS["local"],
        "needs_key": False,
        "hint": "CPU only. Downloads the model on first run and is slower than cloud ASR.",
    },
]


class ProviderError(RuntimeError):
    pass


def _is_local_url(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local")


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.25

    def resolved_provider(self) -> str:
        provider = (self.provider or "groq").strip().lower()
        if provider in ("lmstudio", "vllm", "custom", "local"):
            return "openai"
        return provider

    def resolved_base_url(self) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            base = DEFAULT_BASE_URLS.get(self.resolved_provider(), DEFAULT_BASE_URLS["openai"])
        return base

    def resolved_model(self) -> str:
        model = (self.model or "").strip()
        if not model:
            model = DEFAULT_MODELS.get(self.resolved_provider(), DEFAULT_MODELS["openai"])
        return model


@dataclass
class ASRConfig:
    provider: str = "groq"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    language: str | None = None
    concurrency: int = 0

    def resolved_provider(self) -> str:
        provider = (self.provider or "groq").strip().lower()
        if provider in ("whisper", "whisper-local", "faster-whisper"):
            return "local"
        return provider

    def resolved_base_url(self) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            base = DEFAULT_BASE_URLS.get(self.resolved_provider(), DEFAULT_BASE_URLS["openai"])
        return base

    def resolved_model(self) -> str:
        model = (self.model or "").strip()
        if not model:
            model = DEFAULT_ASR_MODELS.get(self.resolved_provider(), DEFAULT_ASR_MODELS["openai"])
        return model

    def worker_count(self) -> int:
        if self.resolved_provider() == "local":
            return 1
        if self.concurrency > 0:
            return self.concurrency
        return max(1, settings.asr_concurrency)


def _merge_key(config: LLMConfig | ASRConfig) -> str:
    if config.api_key:
        return config.api_key.strip()
    provider = config.resolved_provider()
    if provider == "groq":
        return settings.groq_api_key
    if provider == "gemini":
        return settings.gemini_api_key
    if provider == "openai":
        return settings.openai_api_key
    return ""


def _require_key(config: LLMConfig | ASRConfig) -> str:
    key = _merge_key(config)
    if key:
        return key
    provider = config.resolved_provider()
    where = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider, "an API key")
    raise ProviderError(
        f"No API key for {provider}. Paste a key in Models & Keys, or set {where} in .env. "
        "You can switch to a local model instead."
    )


def _needs_key(config: LLMConfig | ASRConfig) -> bool:
    return config.resolved_provider() in ("groq", "openai", "gemini") and not _is_local_url(
        config.resolved_base_url()
    )


def validate_llm(config: LLMConfig) -> None:
    if _needs_key(config) and not _merge_key(config):
        _require_key(config)


def validate_asr(config: ASRConfig) -> None:
    if _needs_key(config) and not _merge_key(config):
        _require_key(config)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _request(fn, *, label: str, attempts: int = 6) -> httpx.Response:
    delay = 4.0
    for attempt in range(1, attempts + 1):
        try:
            response = fn()
        except httpx.HTTPError as exc:
            if attempt == attempts:
                raise ProviderError(f"{label}: network error: {exc}") from exc
            log(f"  {label}: network error, retrying in {delay:.0f}s ({attempt}/{attempts})")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if response.status_code < 400:
            return response
        retryable = response.status_code in (408, 409, 429, 500, 502, 503, 504)
        if retryable and attempt < attempts:
            wait = _retry_after(response) or delay
            log(f"  {label}: HTTP {response.status_code}, retrying in {wait:.0f}s ({attempt}/{attempts})")
            time.sleep(wait)
            delay = min(delay * 2, 90.0)
            continue
        return response
    raise ProviderError(f"{label}: retries exhausted")


def _check(response: httpx.Response, label: str) -> None:
    if response.status_code >= 400:
        body = " ".join(response.text.split())[:400]
        raise ProviderError(f"{label}: HTTP {response.status_code}: {body}")


def _post(url: str, *, headers: dict, json_body: dict, timeout: float, label: str) -> httpx.Response:
    def call() -> httpx.Response:
        with _client(timeout=timeout) as client:
            return client.post(url, headers=headers, json=json_body)

    return _request(call, label=label)


def _client(**kwargs) -> httpx.Client:
    """httpx 0.28.1 fails to parse bracketed IPv6 entries in NO_PROXY; fall back to no env proxies."""
    try:
        return httpx.Client(**kwargs)
    except httpx.InvalidURL:
        log("  note: ignoring malformed proxy environment (check NO_PROXY)")
        return httpx.Client(trust_env=False, **kwargs)


def _content_text(data: dict) -> str:
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class ChatLLM:
    """OpenAI-compatible chat completions (Groq, OpenAI, OpenRouter, Ollama, LM Studio, vLLM)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.resolved_base_url()
        self.model = config.resolved_model()
        self.api_key = _merge_key(config)
        if _needs_key(config) and not self.api_key:
            _require_key(config)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # Ollama and most local OpenAI-compatible servers ignore the token but some require presence.
            headers["Authorization"] = "Bearer local"
        return headers

    def chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        url = f"{self.base_url}/chat/completions"
        response = _post(url, headers=self._headers(), json_body=payload, timeout=240, label=f"LLM {self.model}")
        if response.status_code in (400, 404, 422):
            # Older local servers reject response_format / temperature. Retry with a minimal body.
            minimal = dict(payload)
            minimal.pop("response_format", None)
            if response.status_code in (400, 422):
                minimal.pop("temperature", None)
            response = _post(url, headers=self._headers(), json_body=minimal, timeout=240, label=f"LLM {self.model}")
        _check(response, f"LLM {self.model}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"LLM {self.model}: response was not JSON: {response.text[:200]}") from exc
        text = _content_text(data)
        if not text:
            raise ProviderError(f"LLM {self.model}: empty response")
        return text


class GeminiLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.resolved_model()
        self.base_url = config.resolved_base_url() or DEFAULT_BASE_URLS["gemini"]
        self.api_key = _merge_key(config) or _require_key(config)

    def chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        generation: dict = {
            "temperature": self.config.temperature,
            "responseMimeType": "application/json",
        }
        if max_tokens:
            generation["maxOutputTokens"] = max_tokens
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        response = _post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json_body=payload,
            timeout=240,
            label=f"Gemini {self.model}",
        )
        _check(response, f"Gemini {self.model}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"Gemini: response was not JSON: {response.text[:200]}") from exc
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        except (KeyError, IndexError, TypeError):
            blocked = data.get("promptFeedback", {}).get("blockReason")
            raise ProviderError(f"Gemini returned no text (blocked={blocked})") from None
        if not text:
            raise ProviderError("Gemini returned no text")
        return text


def make_llm(config: LLMConfig | None = None) -> ChatLLM | GeminiLLM:
    config = config or default_llm_config()
    if config.resolved_provider() == "gemini":
        return GeminiLLM(config)
    return ChatLLM(config)


def _words_from_segments(segments: list[dict]) -> list[dict]:
    """Approximate word timings when the ASR response has no word-level data."""
    words: list[dict] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        tokens = text.split()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if not tokens:
            continue
        span = max(0.05, (end - start) / len(tokens))
        for i, token in enumerate(tokens):
            words.append({"word": token, "start": start + i * span, "end": start + (i + 1) * span})
    return words


class OpenAICompatASR:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.base_url = config.resolved_base_url()
        self.model = config.resolved_model()
        self.api_key = _merge_key(config)
        if _needs_key(config) and not self.api_key:
            _require_key(config)

    def transcribe(self, audio: Path) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key or 'local'}"}
        data: dict = {
            "model": self.model,
            "response_format": "verbose_json",
            "temperature": "0",
            "timestamp_granularities[]": ["word", "segment"],
        }
        if self.config.language:
            data["language"] = self.config.language

        def call() -> httpx.Response:
            with open(audio, "rb") as handle:
                with _client(timeout=600) as client:
                    return client.post(
                        f"{self.base_url}/audio/transcriptions",
                        headers=headers,
                        data=data,
                        files={"file": (audio.name, handle, "audio/mpeg")},
                    )

        response = _request(call, label=f"ASR {self.model}", attempts=7)
        _check(response, f"ASR {self.model}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(f"ASR {self.model}: response was not JSON: {response.text[:200]}") from exc
        segments = payload.get("segments") or []
        words = payload.get("words") or _words_from_segments(segments)
        clean_segments = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")).strip()}
            for s in segments
            if "start" in s and "end" in s
        ]
        clean_words = [
            {"word": str(w.get("word", "")).strip(), "start": float(w["start"]), "end": float(w["end"])}
            for w in words
            if "start" in w and "end" in w and str(w.get("word", "")).strip()
        ]
        return {"words": clean_words, "segments": clean_segments, "language": payload.get("language")}


class LocalWhisperASR:
    def __init__(self, config: ASRConfig):
        self.config = config
        self.model_name = config.resolved_model()
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ProviderError(
                "Local ASR needs faster-whisper. Install it with: pip install faster-whisper"
            ) from exc
        log(f"  loading local whisper model {self.model_name!r} (downloads on first use)")
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, (os.cpu_count() or 4) - 1),
        )
        return self._model

    def transcribe(self, audio: Path) -> dict:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio),
            word_timestamps=True,
            vad_filter=True,
            beam_size=1,
            language=self.config.language,
        )
        words: list[dict] = []
        segments: list[dict] = []
        for segment in segments_iter:
            text = (segment.text or "").strip()
            segments.append({"start": float(segment.start), "end": float(segment.end), "text": text})
            for word in segment.words or []:
                token = (word.word or "").strip()
                if token:
                    words.append({"word": token, "start": float(word.start), "end": float(word.end)})
        if not words:
            words = _words_from_segments(segments)
        return {"words": words, "segments": segments, "language": getattr(info, "language", None)}


def make_asr(config: ASRConfig | None = None) -> OpenAICompatASR | LocalWhisperASR:
    config = config or default_asr_config()
    if config.resolved_provider() == "local":
        return LocalWhisperASR(config)
    return OpenAICompatASR(config)


def default_llm_config() -> LLMConfig:
    provider = (settings.llm_provider or "groq").strip().lower()
    if provider == "gemini":
        return LLMConfig(provider="gemini", model=settings.gemini_llm_model, api_key=settings.gemini_api_key, base_url=settings.gemini_base_url)
    if provider == "ollama":
        return LLMConfig(provider="ollama", model=settings.ollama_llm_model, base_url=settings.ollama_base_url)
    if provider == "openai":
        return LLMConfig(provider="openai", model=settings.openai_llm_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return LLMConfig(provider="groq", model=settings.llm_model, api_key=settings.groq_api_key)


def default_asr_config() -> ASRConfig:
    provider = (settings.asr_provider or "groq").strip().lower()
    if provider == "local":
        return ASRConfig(provider="local", model=settings.local_whisper_model, language=settings.language)
    if provider == "openai":
        return ASRConfig(provider="openai", model=settings.openai_asr_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url, language=settings.language)
    return ASRConfig(provider="groq", model=settings.asr_model, api_key=settings.groq_api_key, base_url=DEFAULT_BASE_URLS["groq"], language=settings.language)


def environment_defaults() -> dict:
    """Describe .env defaults for the UI without ever exposing key values."""
    llm = default_llm_config()
    asr = default_asr_config()
    return {
        "llm": {
            "provider": llm.resolved_provider(),
            "model": llm.resolved_model(),
            "base_url": llm.resolved_base_url(),
            "has_key": bool(_merge_key(llm)),
        },
        "asr": {
            "provider": asr.resolved_provider(),
            "model": asr.resolved_model(),
            "base_url": asr.resolved_base_url(),
            "has_key": bool(_merge_key(asr)),
        },
    }


def provider_catalog() -> dict:
    return {"llm": LLM_PROVIDERS, "asr": ASR_PROVIDERS}


def test_llm(config: LLMConfig) -> dict:
    llm = make_llm(config)
    started = time.time()
    text = llm.chat("You are a connection test.", 'Reply with JSON only: {"ok": true}', max_tokens=64)
    return {
        "ok": True,
        "provider": config.resolved_provider(),
        "model": config.resolved_model(),
        "seconds": round(time.time() - started, 2),
        "sample": text.strip()[:200],
    }


def test_asr(config: ASRConfig, audio: Path) -> dict:
    asr = make_asr(config)
    started = time.time()
    result = asr.transcribe(audio)
    return {
        "ok": True,
        "provider": config.resolved_provider(),
        "model": config.resolved_model(),
        "seconds": round(time.time() - started, 2),
        "language": result.get("language"),
        "words": len(result.get("words") or []),
    }
