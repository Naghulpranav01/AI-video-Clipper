# AI Clipper engine: FastAPI + ffmpeg + yt-dlp + Whisper.
# ffmpeg/libass come from the distro package (static-ffmpeg falls back to it).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OUTPUT_DIR=/data/output

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY clipper ./clipper
COPY .env.example ./
RUN mkdir -p /data/output/uploads

VOLUME ["/data/output"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "clipper", "serve", "--host", "0.0.0.0", "--port", "8000"]
