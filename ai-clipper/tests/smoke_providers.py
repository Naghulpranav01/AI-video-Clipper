"""End-to-end provider test against a fake OpenAI-compatible API.

Verifies the online LLM + ASR HTTP paths, chunk offsetting, parallel render and
the template metadata without needing a real API key.

    .venv/bin/python tests/smoke_providers.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_api import serve  # noqa: E402
from smoke_render import make_source  # noqa: E402

from clipper.pipeline import ClipOptions, clip_video  # noqa: E402
from clipper.providers import ASRConfig, LLMConfig  # noqa: E402
from clipper.util import probe  # noqa: E402

OUT = ROOT / "output" / "provider_test"
SRC = OUT / "src" / "demo.mp4"


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    if not SRC.exists():
        make_source(SRC, 32)

    server, base_url, calls = serve()
    try:
        llm = LLMConfig(provider="openai", model="fake-llm", api_key="test-key", base_url=base_url)
        asr = ASRConfig(provider="openai", model="fake-whisper", api_key="test-key", base_url=base_url, concurrency=2)
        options = ClipOptions(
            count=2,
            min_len=8,
            max_len=20,
            crop="center",
            template="news-box",
            quality="turbo",
            preset="business",
            render_concurrency=2,
            output_dir=OUT,
            llm_config=llm,
            asr_config=asr,
        )
        job = clip_video(str(SRC), options)

        assert len(job["clips"]) == 2, job["clips"]
        assert job["models"] == {"llm": "openai", "llm_model": "fake-llm", "asr": "openai", "asr_model": "fake-whisper"}, job["models"]
        for clip in job["clips"]:
            path = Path(job["workdir"]) / clip["file"]
            assert path.exists() and path.stat().st_size > 10_000, path
            info = probe(path)
            assert (info["width"], info["height"]) == (720, 1280), info
            assert clip["template"] == "news-box" and clip["quality"] == "turbo", clip

        transcript = json.loads((Path(job["workdir"]) / "transcript.json").read_text())
        assert len(transcript["words"]) > 50, len(transcript["words"])

        kinds = [c[0] for c in calls]
        assert "asr" in kinds and "chat" in kinds, kinds
        assert all(c[1] == "Bearer test-key" for c in calls), calls
        print(f"provider pipeline ok: {len(job['clips'])} clips, {len(transcript['words'])} words, {len(calls)} API calls")
        print(json.dumps({"ok": True, "workdir": job["workdir"]}, indent=2))
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
