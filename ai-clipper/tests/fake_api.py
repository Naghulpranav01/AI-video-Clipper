"""A tiny OpenAI-compatible server for offline provider tests.

Serves /v1/chat/completions and /v1/audio/transcriptions with deterministic
fixtures so the whole pipeline can run without an API key or network.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PHRASE = (
    "most people price their product by looking at competitors and guessing "
    "but the real number comes from the value the buyer gets not what the market charges "
    "so ask what changes for them after they buy and price against that outcome "
    "that is how you stop competing on price and start competing on results"
).split()


def _words(duration: float = 34.0) -> list[dict]:
    step = duration / len(PHRASE)
    return [
        {"word": word, "start": round(i * step, 3), "end": round((i + 1) * step, 3)}
        for i, word in enumerate(PHRASE)
    ]


def _segments(words: list[dict], size: int = 8) -> list[dict]:
    segments = []
    for i in range(0, len(words), size):
        chunk = words[i : i + size]
        segments.append(
            {
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "text": " ".join(w["word"] for w in chunk),
            }
        )
    return segments


def moments_payload() -> dict:
    return {
        "moments": [
            {
                "start": 2.0,
                "end": 14.0,
                "title": "Price against the outcome",
                "hook": "You are pricing for the wrong person",
                "score": 91,
                "reason": "Contrarian claim with a concrete fix.",
            },
            {
                "start": 17.0,
                "end": 29.0,
                "title": "Stop competing on price",
                "hook": "Compete on results instead",
                "score": 84,
                "reason": "Actionable close with a clear payoff.",
            },
        ]
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # silence
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        auth = self.headers.get("Authorization", "")
        if self.path.endswith("/chat/completions"):
            self.server.calls.append(("chat", auth, self.path))
            content = json.dumps(moments_payload())
            return self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})
        if self.path.endswith("/audio/transcriptions"):
            self.server.calls.append(("asr", auth, self.path))
            words = _words()
            return self._send(
                200,
                {
                    "language": "en",
                    "duration": words[-1]["end"],
                    "segments": _segments(words),
                    "words": words,
                },
            )
        return self._send(404, {"error": "not found"})


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.calls = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/v1", server.calls
