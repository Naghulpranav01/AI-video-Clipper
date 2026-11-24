#!/usr/bin/env bash
# One-command start: creates .venv, installs deps, starts the web UI.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
. .venv/bin/activate

python -c "import fastapi, faster_whisper" 2>/dev/null || pip install -r requirements.txt
[ -f .env ] || cp .env.example .env

exec python -m clipper serve --host "$HOST" --port "$PORT"
