# AI Clipper — self-hosted long-video → shorts pipeline

Turn hour-long podcasts, webinars, streams and interviews into captioned 9:16 clips with hooks and virality scores. Runs on a plain CPU machine with a free Groq key, or fully offline with local models. No watermark, no minute limits, no subscription.

This is an independent, clean-room implementation inspired by the workflow of [OpenShorts](https://github.com/mutonby/openshorts) (MIT). It does not contain or modify OpenShorts code, and it does not bypass anything: OpenShorts self-hosting is already free and unlimited, while its paid cloud tier is mainly their GPU, their API keys and their always-on Autopilot. This project gives you the same clipping pipeline on your own machine, with your own free key or local models.

## What you get

- **Any long source**: YouTube/URL via `yt-dlp`, or a local file. Hours-long sources work; audio is chunked and transcribed in parallel.
- **Online or local models**:
  - LLM: Groq, any OpenAI-compatible API (OpenAI, OpenRouter, Together, DeepSeek, LM Studio, vLLM), Google Gemini, or Ollama on your machine.
  - ASR: Groq Whisper, any OpenAI-compatible `/audio/transcriptions` server, or local `faster-whisper` (CPU, fully offline).
- **Keys in the UI**: a *Models & Keys* tab with separate **Online** and **Local** model tabs, API key fields, model/base-URL overrides and one-click connection tests. Keys live in your browser's localStorage and are sent only to this local server — never written to disk or logged. `.env` remains supported as a server default.
- **Six caption templates** with live 9:16 previews: Bold Pop, Clean White, Karaoke Green, Beast Mode, Minimal, News Box. Each controls fonts, colours, word grouping, highlight, caption box, hook banner and hook size.
- **Seven clip-focus presets** for moment picking: general, podcast, interview, tutorial, comedy, business, motivation.
- **Viral moment detection**: the LLM reads the timestamped transcript window by window, returns self-contained 15–60s moments with a 0–100 score, title, on-screen hook and rationale. Overlapping candidates are removed and timestamps snap to sentence bounds.
- **9:16 reframing**: face-tracked crop that pans with the speaker (OpenCV), blurred-background mode for wide/two-shot footage, or a fixed center crop.
- **Burned captions + hook**: word-timed phrases, highlighted current word, styled opening hook, SRT/ASS export.
- **Fast path for long videos**: Turbo 720p preset (ultrafast, ~2× faster than 1080p), 720p-capped downloads, parallel audio chunks, parallel clip rendering, dynaudnorm instead of the slower loudnorm, and cached transcripts/moments.
- **Deliverables per job**: `clip_01/clip.mp4`, `.srt`, `.ass`, `thumbnail.jpg`, `meta.json`, plus `job.json` and a browsable `gallery.html`.
- **Cheap re-edits**: `rerender` reuses the transcript and moments to change template/framing/captions without paying for transcription again.
- **Local web UI** and CLI, plus optional auto-posting through Upload-Post.

## Feature parity with the OpenShorts cloud paid tier

| Capability | Here | Notes |
|---|---|---|
| Viral moment detection | Yes | Any online LLM or local Ollama/LM Studio model |
| Smart 9:16 reframing | Yes | Tracked single speaker, blurred background, center |
| Auto captions + hooks | Yes | Six visual templates, word-level highlight, SRT/ASS |
| Model selection | Yes | Online/local tabs, BYO keys, connection tests |
| No watermark / no cap | Yes | Self-hosted by design |
| Batch clips per video | Yes | Up to 20 clips, rendered in parallel |
| Clip editor | Partial | `rerender` with new template/crop/captions/hook; no timeline UI |
| Auto-posting | Yes, BYO key | Upload-Post paid plan for TikTok |
| Autopilot (channel watcher) | No | Run `clip` on a cron/queue for the same effect |
| AI UGC actor videos / YouTube Studio | No | Different product surface |

## Requirements

- Python 3.11+ (tested on 3.12)
- 4 GB RAM, any 4-core CPU. No GPU needed.
- Internet for `yt-dlp`, cloud APIs, the first-run ffmpeg download and the first local-whisper model download.
- A free Groq key (<https://console.groq.com/keys>), another online key, **or** local models (no key at all).

ffmpeg (with libx264 + libass) is installed automatically by `static-ffmpeg` — no sudo, no system packages. `faster-whisper` is included for local ASR; `av` is pinned below 16 because newer PyAV releases broke faster-whisper's decoder call at the time of writing.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Option A: use the web UI (recommended)
.venv/bin/python -m clipper serve
# open http://127.0.0.1:8765 → Models & Keys → paste a Groq key → Test LLM

# Option B: .env + CLI
cp .env.example .env
# edit .env and paste GROQ_API_KEY=gsk_...
.venv/bin/python -m clipper clip "https://www.youtube.com/watch?v=..." -n 6 --template beast --quality turbo
```

Output lands in `output/<video-name>-<timestamp>/`:

```
job.json                 # source metadata, options, models, moments, clip list
gallery.html             # open in a browser, or serve with `clipper serve`
transcript.json          # word-level transcript (reused by rerender)
moments.json             # editable moment list
clip_01/clip.mp4         # 720x1280 (turbo) or 1080x1920, captions + hook burned in
clip_01/captions.srt
clip_01/thumbnail.jpg
```

### Web UI

```bash
.venv/bin/python -m clipper serve          # http://127.0.0.1:8765
```

- **Create tab**: paste a URL or upload a file, pick a template card, clip focus, render quality, framing, clip count/length, parallel renders, then generate. Live log and preview grid.
- **Models & Keys tab**:
  - **Online**: Groq / OpenAI-compatible / Gemini for the moment picker; Groq or OpenAI-compatible for Whisper. Paste keys, override base URLs/models, press **Test**.
  - **Local**: Ollama or LM Studio for the moment picker, `faster-whisper` for transcription, with the same test buttons. No keys needed.
- Keys are saved in this browser's `localStorage` and sent to the local server per job. The server never persists or logs them. Blank fields fall back to `.env`.
- Keep the server bound to `127.0.0.1`: it can read local paths and receives your API keys.

### Templates

| Key | Look |
|---|---|
| `bold-pop` | White caps, yellow karaoke word, yellow hook banner (default) |
| `clean-white` | Sentence-case white captions, no colour pop, outlined hook |
| `karaoke-green` | Tight 3-word phrases, green highlight, yellow banner |
| `beast` | Huge yellow text, thick black outline, red banner |
| `minimal` | Small lowercase captions low in frame, light outlined hook |
| `news-box` | Boxed red caption bars, white text, red headline banner |

Templates are pure ASS styles in [clipper/templates.py](clipper/templates.py); add another dataclass entry to get a new one in the UI, CLI and API.

### Models

Online (fastest for multi-hour videos):

```bash
# Groq default
.venv/bin/python -m clipper clip video.mp4

# OpenAI-compatible (OpenRouter example)
export OPENAI_API_KEY=sk-or-...
.venv/bin/python -m clipper clip video.mp4 \
  --llm-provider openai --llm-base-url https://openrouter.ai/api/v1 \
  --llm-model anthropic/claude-3.5-sonnet --asr-provider groq
```

Local (no keys, no cloud):

```bash
ollama pull llama3.1:8b
.venv/bin/python -m clipper clip video.mp4 \
  --llm-provider ollama --llm-model llama3.1:8b \
  --asr-provider local --asr-model base   # tiny | base | small | medium
```

Local Whisper runs on CPU and is usually the bottleneck for hours-long sources; cloud ASR at 10-minute chunks is much faster. Local LLMs are fine for the transcript windows, but slower.

### CLI

```bash
python -m clipper clip URL_OR_FILE [-n 5] [--min 15] [--max 60]
                        [--crop track|blur|center]
                        [--template bold-pop|clean-white|karaoke-green|beast|minimal|news-box]
                        [--preset general|podcast|interview|tutorial|comedy|business|motivation]
                        [--quality standard|turbo] [--jobs N]
                        [--llm-provider groq|openai|gemini|ollama] [--llm-model M] [--llm-base-url U]
                        [--asr-provider groq|openai|local] [--asr-model M] [--asr-base-url U]
                        [--no-captions] [--no-hook]
                        [--moments moments.json] [--transcript transcript.json]
                        [--force-transcribe] [--out DIR]
python -m clipper rerender WORKDIR [--template ...] [--quality ...] [--crop ...] [--jobs N]
                        [--no-captions] [--no-hook] [--moments FILE]
python -m clipper transcribe URL_OR_FILE [--asr-provider ...] [--asr-model ...]
python -m clipper serve [--host 127.0.0.1] [--port 8765]
python -m clipper publish CLIP.mp4 --title "..." [--platforms tiktok,instagram,youtube]
```

## How it works

```
ingest (yt-dlp/local, height capped by quality preset)
  └─ ffmpeg: 10-min mono 16 kHz mp3 chunks (parallel extraction)
       └─ ASR provider (Groq/OpenAI-compatible/local whisper), parallel chunks
            └─ words + segments with offsets, deduped at chunk seams
                 └─ blocks → ~18k-char windows
                      └─ LLM provider (online/local) + content preset → candidate moments
                           └─ snap to sentence bounds, dedupe overlaps, rank, top N
                                └─ per clip (N workers in parallel):
                                     cut + scale to 720x1280 or 1080x1920
                                     OpenCV Haar face sampling (2 fps) → median-smoothed
                                       piecewise crop-x expression (pans with the speaker)
                                     template ASS captions + hook → libass burn-in
                                     dynaudnorm → H.264/AAC → mp4 + thumb + SRT
```

Key modules: `clipper/providers.py` (all HTTP/local model adapters), `templates.py`, `ingest.py`, `transcribe.py`, `moments.py`, `faces.py`, `subtitles.py`, `render.py`, `pipeline.py`, `web.py`.

## Cost and speed

Measured on this test VM (4 vCPU, 5 GB RAM, no GPU), 10-second clip, `bold-pop`:

| Mode | Render time | Effective speed | Output |
|---|---|---|---|
| `--quality turbo` | ~15–17 s | ~0.6× realtime | 720×1280, ultrafast CRF 23 |
| `--quality standard` | ~34 s | ~0.3× realtime | 1080×1920, veryfast CRF 20 |

- Turbo is roughly **2× faster** than standard and also caps the download at 720p, which matters for multi-hour sources.
- Two parallel clips overlap face tracking/analysis and improve throughput ~10–25%; the H.264 encode itself is CPU-bound, so more parallel jobs mostly trade per-clip speed for total throughput. Keep it at 2 on 4 cores.
- Example end-to-end estimate for a 2-hour podcast producing 8 clips × 45 s (6 minutes of output) on this VM: download + transcription + moment picking a few minutes, then **~10 minutes of turbo rendering** (vs ~25+ minutes before the turbo path). A modern desktop is 2–4× faster.
- Cloud transcription is cheap on Groq's free tier. 429s are retried with backoff honoring `retry-after`. Reduce `ASR_CONCURRENCY` if you hit TPM limits.
- Local Whisper and local LLMs have zero API cost but use CPU: expect roughly realtime (tiny/base, depending on cores) for transcription and several seconds per transcript window for an 8B model.
- Templates differ slightly in cost: `news-box` renders heavier due to boxed libass events; `center` framing is the cheapest (no face detection).

## Configuration

Everything is optional; see [.env.example](.env.example). The UI can override providers, keys and models per job.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | default moment picker: `groq`, `openai`, `gemini`, `ollama` |
| `ASR_PROVIDER` | `groq` | default transcription: `groq`, `openai`, `local` |
| `GROQ_API_KEY` | — | Groq key (LLM + Whisper) |
| `GROQ_LLM_MODEL` / `GROQ_ASR_MODEL` | `llama-3.3-70b-versatile` / `whisper-large-v3-turbo` | Groq models |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — / `https://api.openai.com/v1` | OpenAI-compatible LLM/ASR |
| `OPENAI_LLM_MODEL` / `OPENAI_ASR_MODEL` | `gpt-4o-mini` / `whisper-1` | OpenAI-compatible models |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Gemini |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://127.0.0.1:11434/v1` / `llama3.1:8b` | local LLM |
| `LOCAL_WHISPER_MODEL` | `base` | local ASR size |
| `ASR_LANGUAGE` | auto | force a language code, e.g. `en` |
| `ASR_CHUNK_SECONDS` | `600` | audio chunk size |
| `ASR_CONCURRENCY` | `3` | parallel cloud transcription requests |
| `LLM_CHAR_BUDGET` | `18000` | transcript chars per LLM call |
| `OUTPUT_DIR` | `output` | job output root |
| `MAX_SOURCE_HEIGHT` | `1080` | max download height (turbo caps at 720) |
| `TEMPLATE` | `bold-pop` | default caption/hook template |
| `RENDER_QUALITY` | `standard` | `standard` or `turbo` |
| `RENDER_CONCURRENCY` | `2` | clips rendered in parallel |
| `HOOK_SECONDS` | `3.6` | how long the hook stays |
| `UPLOAD_POST_API_KEY` / `UPLOAD_POST_USER` | — | optional publishing |

## Publishing

```bash
export UPLOAD_POST_API_KEY=...
export UPLOAD_POST_USER=my-profile

.venv/bin/python -m clipper publish output/job/clip_01/clip.mp4 \
  --title "You are pricing for the wrong person" \
  --platforms tiktok,instagram,youtube
```

Upload-Post runs its own approved apps, so you connect accounts once in their dashboard. TikTok needs a paid Upload-Post plan. Endpoint and fields follow <https://docs.upload-post.com/api/upload-video>.

## Deployment

### Option 0: one command on any machine

```bash
./run.sh          # creates .venv, installs deps, serves http://127.0.0.1:8765
HOST=0.0.0.0 PORT=8000 ALLOW_ORIGINS='*' ./run.sh   # reachable from other machines
```

### Option 1: Docker on any VM (full app, recommended)

Vercel/serverless cannot run this engine: rendering needs ffmpeg and minutes per clip, jobs run for tens of minutes on multi-hour videos, and output files must persist. A small VM or container host is the right home for the engine.

```bash
cp .env.example .env            # add GROQ_API_KEY or switch to local models
docker compose up -d --build    # http://<server>:8000
```

`output/` is mounted to `/data/output` so clips survive restarts. Put Caddy/nginx in front for HTTPS, set `ALLOW_ORIGINS` to your UI's origin (CORS is off by default), and remember the API accepts client-supplied keys.

### Option 2: UI on Vercel, engine on a server

Use the `ai-clipper-vercel/` folder (in the dist bundle) as a static site on Vercel. It is the same UI; set the engine URL once in **Models & Keys → Clipper server** (or open the site with `?server=https://clips.example.com`). All API calls, uploads and clip playback then go straight to your engine.

What Vercel can and cannot do:

| | Vercel |
|---|---|
| Serve the UI | yes — static, free tier |
| Run uploads/render/transcribe | no — 60–300s function limit, /tmp-only disk, 250 MB bundle limit |
| Persistent clips | no — use the engine host |

The engine needs a public HTTPS URL the browser can reach (a VM with a domain, or a tunnel such as `cloudflared tunnel --url http://127.0.0.1:8765` for a quick test). Because the UI page is HTTPS, the engine must also be HTTPS to avoid mixed-content blocking. Start the engine with `--host 0.0.0.0` and set `ALLOW_ORIGINS=*` (or your Vercel domain) in `.env`; without it the browser cannot call the API.

## Tests (no API key needed)

```bash
.venv/bin/python tests/smoke_render.py       # all 3 framing modes + every template (ASS)
.venv/bin/python tests/smoke_tracking.py     # pans a face across the frame, checks tracking
.venv/bin/python tests/smoke_providers.py    # full pipeline against a fake OpenAI-compatible API
```

`SMOKE_QUALITY=turbo` makes the render test exercise the 720p path. `smoke_providers.py` starts a local fake server, so it validates the HTTP LLM/ASR code, chunk offsetting, auth headers, template/quality metadata and parallel rendering without any key.

## Troubleshooting

- **"No API key for ..."** → paste a key in *Models & Keys* (online tab) or set it in `.env`, or switch to local models. A blank UI key falls back to `.env`.
- **Local LLM connection refused** → start Ollama (`ollama serve`) or LM Studio's server, pull a model (`ollama pull llama3.1:8b`), and match the base URL (`http://127.0.0.1:11434/v1` or `http://127.0.0.1:1234/v1`).
- **First local-whisper run is slow** → the model downloads on first use (tiny ~75 MB, base ~145 MB). It is cached afterwards.
- **Local Whisper decoder error (`metadata_errors`)** → PyAV is too new for your faster-whisper; `pip install "av>=11,<16"` (pinned in requirements).
- **No faces detected** → the crop falls back to center. Use `--crop blur` for two-shots or wide stages, where a single tracked crop would cut people off.
- **Captions missing** → install `DejaVuSans-Bold.ttf` or set `CAPTION_FONT` to an installed family; check `fc-list | grep -i dejavu`.
- **Groq 429** → free-tier TPM limits; the code retries with backoff and honors `retry-after`. Reduce `ASR_CONCURRENCY`, or use a smaller LLM (`GROQ_LLM_MODEL=llama-3.1-8b-instant`).
- **Slow renders** → use `--quality turbo`, `--crop center`, `--jobs 2`, fewer clips, or shorter max length.
- **Corrupt download / no video stream** → delete the job folder and retry; update yt-dlp with `.venv/bin/pip install -U yt-dlp`.
- **OpenCV** → pinned `<5` because OpenCV 5 removed the legacy Haar cascade API used for tracking.
- **Keys** → keys are sent from the browser to the local server over localhost and kept only in memory for the job. Do not expose the server publicly or run it behind a proxy that logs request bodies.

## Notes on the original request

The OpenShorts repo is MIT and self-hosting it is free and uncapped, so there is no local paywall to remove. The paid tier is the hosted service (GPU rendering, managed Gemini key, always-on Autopilot) — that cannot be "unlocked" from the code, only paid for or rebuilt. This project rebuilds the useful part with your own key or local models. If you want the full OpenShorts dashboard/editor/AI-shorts features, run their repo self-hosted with Docker; the two can coexist.
