from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .util import FFMPEG, log, run


def _detect_frames(source: Path, start: float, duration: float, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    run(
        [
            FFMPEG, "-y", "-v", "error",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{max(0.1, duration):.3f}",
            "-i", source,
            "-vf", "fps=2,scale=640:-2",
            "-q:v", "4",
            frames_dir / "f_%05d.jpg",
        ],
        capture=True,
    )
    return sorted(frames_dir.glob("f_*.jpg"))


def _median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = np.median(padded[i : i + window])
    return out


def face_track(source: Path, start: float, duration: float, workdir: Path) -> list[tuple[float, float]] | None:
    """Return (seconds_into_clip, normalized_x) crop centers for the dominant speaker."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        log("  face detector unavailable, using center crop")
        return None

    frames = _detect_frames(source, start, duration, workdir / "frames")
    if not frames:
        return None

    detections: list[list[tuple[float, float, float]]] = []  # per frame: (norm_x, norm_area, area)
    for frame_path in frames:
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            detections.append([])
            continue
        faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        height, width = image.shape[:2]
        found = []
        for x, y, w, h in faces:
            found.append(((x + w / 2) / width, (w * h) / (width * height), float(w * h)))
        detections.append(found)

    total_area = sum(area for frame in detections for _x, _a, area in frame)
    if total_area <= 0:
        log("  no faces detected, using center crop")
        return None

    # Lock onto the speaker with the most screen presence across the clip.
    xs = [norm_x for frame in detections for norm_x, _a, _area in frame]
    areas = [area for frame in detections for _x, _a, area in frame]
    dominant_x = float(np.average(xs, weights=areas)) if len(xs) > 1 else xs[0]

    samples: list[tuple[float, float]] = []
    missing: list[float] = []
    for i, frame in enumerate(detections):
        t = i * 0.5
        if not frame:
            missing.append(t)
            continue
        best = min(frame, key=lambda item: abs(item[0] - dominant_x))
        if abs(best[0] - dominant_x) > 0.35:
            missing.append(t)
            continue
        samples.append((t, best[0]))

    if len(samples) < max(2, len(frames) * 0.1):
        log("  face signal too weak, using center crop")
        return None

    grid = np.arange(0.0, max(duration, 0.1) + 1e-9, 0.5)
    times = np.array([s[0] for s in samples])
    values = np.array([s[1] for s in samples])
    resampled = np.interp(grid, times, values)
    smoothed = _median_filter(resampled, window=7)

    if len(smoothed) > 3:
        # second pass kills any remaining jump between speakers
        smoothed = _median_filter(smoothed, window=5)
    log(f"  tracked face in {len(samples)}/{len(frames)} samples, dominant x={dominant_x:.2f}")
    return [(float(t), float(max(0.0, min(1.0, x)))) for t, x in zip(grid, smoothed)]


def keyframe_crop_x(samples: list[tuple[float, float]], duration: float, scaled_w: int, out_w: int) -> str:
    """Build a piecewise-linear ffmpeg crop-x expression from (t, norm_x) samples."""
    limit = scaled_w - out_w
    if limit <= 0:
        return "0"
    step = max(1.0, duration / 60.0)  # keep the expression to ~60 ifs
    count = max(1, int(math.ceil(duration / step)))
    keyframes: list[tuple[float, int]] = []
    for i in range(count + 1):
        t = min(float(i) * step, duration)
        norm_x = _value_at(samples, t)
        x = int(round(norm_x * scaled_w - out_w / 2))
        keyframes.append((round(t, 2), max(0, min(limit, x))))
    keyframes = _dedupe_keyframes(keyframes)
    if not keyframes:
        return str(limit // 2)
    if len({x for _t, x in keyframes}) == 1:
        return str(keyframes[0][1])

    def build(i: int) -> str:
        t, x = keyframes[i]
        if i == len(keyframes) - 1:
            return str(x)
        t_next, x_next = keyframes[i + 1]
        span = max(0.01, t_next - t)
        segment = f"{x}+({x_next}-{x})*(t-{t})/{round(span, 2)}"
        return f"if(lt(t,{t_next}),{segment},{build(i + 1)})"

    return build(0)


def _value_at(samples: list[tuple[float, float]], t: float) -> float:
    if t <= samples[0][0]:
        return samples[0][1]
    if t >= samples[-1][0]:
        return samples[-1][1]
    for (t0, x0), (t1, x1) in zip(samples, samples[1:]):
        if t0 <= t <= t1:
            if t1 - t0 < 1e-9:
                return x1
            ratio = (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * ratio
    return samples[-1][1]


def _dedupe_keyframes(keyframes: list[tuple[float, int]]) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    for t, x in keyframes:
        if result and t <= result[-1][0] + 1e-6:
            continue
        if len(result) >= 2 and result[-1][1] == x and result[-2][1] == x:
            continue
        result.append((t, x))
    return result
