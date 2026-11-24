from __future__ import annotations

from pathlib import Path

import httpx

from .config import settings
from .util import log

UPLOAD_URL = "https://api.upload-post.com/api/upload"


def publish_clip(
    video: Path,
    platforms: list[str],
    *,
    title: str,
    description: str = "",
    user: str | None = None,
    scheduled_date: str | None = None,
) -> dict:
    """Post a rendered clip through Upload-Post (TikTok, Instagram, YouTube, ...).

    Upload-Post needs a paid plan for TikTok and your own connected profile; see
    https://docs.upload-post.com/api/upload-video
    """
    key = settings.upload_post_key
    profile = user or settings.upload_post_user
    if not key:
        raise SystemExit("UPLOAD_POST_API_KEY is not set (get one at https://app.upload-post.com)")
    if not profile:
        raise SystemExit("set UPLOAD_POST_USER (or pass --user) to the Upload-Post profile name")
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    data: list[tuple[str, str]] = [("user", profile), ("title", title)]
    if description:
        data.append(("description", description))
    if scheduled_date:
        data.append(("scheduled_date", scheduled_date))
    for platform in platforms:
        data.append(("platform[]", platform))

    log(f"publishing {video.name} to {', '.join(platforms)} as '{profile}'")
    with open(video, "rb") as handle:
        response = httpx.post(
            UPLOAD_URL,
            headers={"Authorization": f"Apikey {key}"},
            data=data,
            files={"video": (video.name, handle, "video/mp4")},
            timeout=180.0,
        )
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text[:2000]}
    if response.status_code >= 400:
        raise RuntimeError(f"Upload-Post error {response.status_code}: {payload}")
    log(f"  Upload-Post response: {payload}")
    return payload
