# AI Clipper — Vercel frontend

This folder is a static copy of the AI Clipper UI. Vercel serves the page; the
clipping engine (ffmpeg, yt-dlp, Whisper) runs on your own machine/VM and the UI
talks to it directly.

## Why the engine is not on Vercel

Rendering burns captions with ffmpeg for tens of seconds per clip, multi-hour
videos take tens of minutes, and clips must be stored on disk. Vercel functions
have a 60–300 s limit, a read-only filesystem (only `/tmp`), and a 250 MB
deployment cap — so jobs cannot run there. The engine is a normal long-running
Python service: use the `ai-clipper` folder (Docker or `run.sh`).

## Deploy the UI

```bash
cd vercel-ui
npx vercel deploy --prod
```

Or push this repo and import it in the Vercel dashboard with **Root Directory**
set to `vercel-ui`. No build step is needed.

## Connect it to the engine

1. Start the engine somewhere reachable over **HTTPS** (Vercel is HTTPS, so plain
   HTTP gets blocked as mixed content):
   - VM: `docker compose up -d` behind Caddy/nginx with a domain; or
   - Quick tunnel: `cloudflared tunnel --url http://127.0.0.1:8765`
2. Enable CORS for the UI origin (off by default): add
   `ALLOW_ORIGINS=https://your-app.vercel.app` (or `*`) to the engine's `.env`.
3. Open the Vercel site → **Models & Keys → Clipper server** → paste the engine
   URL (e.g. `https://clips.example.com`) → press Tab/Enter to save.

Shortcut: `https://your-app.vercel.app/?server=https://clips.example.com`
configures it in one click. You can also bake the URL into `config.js` and
redeploy.

## Notes

- API keys are typed into the browser and stored only in its `localStorage`.
  They are sent to your engine with each job, never to Vercel.
- Uploads go browser → engine directly, so large files are not limited by
  Vercel's 4.5 MB function body cap.
- The engine URL is also saved in the browser, so each visitor/configures once.
- If the page shows "Could not reach the clipper server", the engine URL is
  wrong, the engine is http (mixed content), or CORS is disabled
  (`ALLOW_ORIGINS` empty).
