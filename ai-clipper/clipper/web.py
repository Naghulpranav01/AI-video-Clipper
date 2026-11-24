from __future__ import annotations

import math
import shutil
import uuid
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .moments import CONTENT_PRESETS, DEFAULT_PRESET, preset_catalog
from .pipeline import ClipOptions, clip_video
from .providers import (
    ASRConfig,
    LLMConfig,
    ProviderError,
    default_asr_config,
    default_llm_config,
    environment_defaults,
    provider_catalog,
    test_asr,
    test_llm,
    validate_asr,
    validate_llm,
)
from .render import DEFAULT_QUALITY, QUALITIES, quality_catalog
from .templates import DEFAULT_TEMPLATE, TEMPLATES, template_catalog
from .util import log

OUTPUT_ROOT = Path(settings.output_dir).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = OUTPUT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Clipper", version="0.2.1")
_origins = [origin.strip() for origin in settings.allow_origins.split(",") if origin.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=600,
    )
app.mount("/files", StaticFiles(directory=OUTPUT_ROOT), name="files")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-job")


@dataclass
class Job:
    id: str
    input: str
    status: str = "queued"
    logs: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    result: dict | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}


class ModelRequest(BaseModel):
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


class JobRequest(BaseModel):
    input: str = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=20)
    min_len: float = Field(default=15.0, ge=5.0, le=180.0)
    max_len: float = Field(default=60.0, ge=10.0, le=300.0)
    crop: str = Field(default="track", pattern="^(track|blur|center)$")
    captions: bool = True
    hook: bool = True
    template: str = DEFAULT_TEMPLATE
    preset: str = DEFAULT_PRESET
    quality: str = DEFAULT_QUALITY
    render_concurrency: int = Field(default=2, ge=1, le=4)
    llm: ModelRequest | None = None
    asr: ModelRequest | None = None


class TestRequest(BaseModel):
    kind: str = Field(pattern="^(llm|asr)$")
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


def _progress(job: Job):
    def emit(message: str) -> None:
        job.logs.append(message)
        if len(job.logs) > 500:
            del job.logs[:100]
        log(message)

    return emit


def _resolved_llm(request: JobRequest) -> LLMConfig:
    if request.llm is None:
        return default_llm_config()
    return LLMConfig(
        provider=request.llm.provider or settings.llm_provider,
        model=request.llm.model,
        api_key=request.llm.api_key,
        base_url=request.llm.base_url,
    )


def _resolved_asr(request: JobRequest, llm: LLMConfig) -> ASRConfig:
    if request.asr is None:
        config = default_asr_config()
    else:
        config = ASRConfig(
            provider=request.asr.provider or settings.asr_provider,
            model=request.asr.model,
            api_key=request.asr.api_key,
            base_url=request.asr.base_url,
            language=settings.language,
        )
    # one Groq/OpenAI key is usually enough for both models
    if not config.api_key and request.llm and request.llm.api_key:
        if config.resolved_provider() == llm.resolved_provider() != "local":
            config.api_key = request.llm.api_key
    return config


def _run(job: Job, request: JobRequest) -> None:
    job.status = "running"
    try:
        llm = _resolved_llm(request)
        asr = _resolved_asr(request, llm)
        options = ClipOptions(
            count=request.count,
            min_len=request.min_len,
            max_len=request.max_len,
            crop=request.crop,
            captions=request.captions,
            hook=request.hook,
            template=request.template,
            quality=request.quality,
            preset=request.preset,
            render_concurrency=request.render_concurrency,
            output_dir=OUTPUT_ROOT,
            llm_config=llm,
            asr_config=asr,
        )
        job.result = clip_video(request.input, options, progress=_progress(job))
        job.status = "done"
    except Exception as exc:
        job.error = str(exc)
        job.status = "error"
        job.logs.append(f"ERROR: {exc}")


def _rel(workdir: str) -> str:
    return Path(workdir).resolve().relative_to(OUTPUT_ROOT).as_posix()


def _public(job: Job) -> dict:
    payload = {
        "id": job.id,
        "input": job.input,
        "status": job.status,
        "created": job.created,
        "logs": job.logs,
        "error": job.error,
        "result": None,
    }
    if job.result:
        rel = _rel(job.result["workdir"])
        result = dict(job.result)
        result["workdir_rel"] = rel
        result["gallery_url"] = f"/files/{rel}/gallery.html"
        result["clips"] = [
            {
                **clip,
                "url": f"/files/{rel}/{clip['file']}",
                "poster_url": f"/files/{rel}/{clip['thumbnail']}",
                "srt_url": f"/files/{rel}/{clip['captions']}",
            }
            for clip in job.result["clips"]
        ]
        payload["result"] = result
    return payload


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).with_name("ui.html")).read_text(encoding="utf-8")


@app.get("/api/catalog")
def catalog() -> dict:
    defaults = environment_defaults()
    defaults.update(
        {
            "template": settings.render_template if settings.render_template in TEMPLATES else DEFAULT_TEMPLATE,
            "quality": settings.render_quality if settings.render_quality in QUALITIES else DEFAULT_QUALITY,
            "preset": DEFAULT_PRESET,
            "render_concurrency": settings.render_concurrency,
        }
    )
    return {
        "templates": template_catalog(),
        "presets": preset_catalog(),
        "qualities": quality_catalog(),
        "providers": provider_catalog(),
        "defaults": defaults,
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "groq_key": bool(settings.groq_api_key),
        "model": default_llm_config().resolved_model(),
        "defaults": environment_defaults(),
    }


@app.post("/api/test")
def test_model(request: TestRequest) -> dict:
    try:
        if request.kind == "llm":
            config = LLMConfig(
                provider=request.provider or settings.llm_provider,
                model=request.model,
                api_key=request.api_key,
                base_url=request.base_url,
            )
            validate_llm(config)
            return test_llm(config)
        config = ASRConfig(
            provider=request.provider or settings.asr_provider,
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            language=settings.language,
        )
        validate_asr(config)
        return test_asr(config, _test_audio())
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload.mp4").name
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}-{name}"
    with open(target, "wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    return {"path": str(target), "name": name}


@app.post("/api/jobs")
def create_job(request: JobRequest) -> dict:
    if request.template not in TEMPLATES:
        raise HTTPException(status_code=422, detail=f"unknown template {request.template!r}")
    if request.preset not in CONTENT_PRESETS:
        raise HTTPException(status_code=422, detail=f"unknown preset {request.preset!r}")
    if request.quality not in QUALITIES:
        raise HTTPException(status_code=422, detail=f"unknown quality {request.quality!r}")
    llm = _resolved_llm(request)
    asr = _resolved_asr(request, llm)
    try:
        validate_llm(llm)
        validate_asr(asr)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, input=request.input)
    JOBS[job_id] = job
    executor.submit(_run, job, request)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)[:20]
    return [
        {
            "id": j.id,
            "input": j.input,
            "status": j.status,
            "created": j.created,
            "clips": len(j.result["clips"]) if j.result else 0,
            "workdir_rel": _rel(j.result["workdir"]) if j.result else None,
        }
        for j in jobs
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _public(job)


@app.get("/api/jobs/{job_id}/gallery")
def gallery(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="job not finished")
    return FileResponse(Path(job.result["workdir"]) / "gallery.html")


_TEST_AUDIO = UPLOAD_DIR / "_asr_test.wav"


def _test_audio() -> Path:
    if _TEST_AUDIO.exists():
        return _TEST_AUDIO
    rate = 16000
    samples = array("h", (int(5000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(rate * 2)))
    with wave.open(str(_TEST_AUDIO), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return _TEST_AUDIO


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    print(f"\nAI Clipper UI: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
