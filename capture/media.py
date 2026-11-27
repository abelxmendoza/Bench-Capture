"""Phone / MacBook clip probing via ffprobe + ffmpeg (no Python video deps)."""
from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

_SHOWINFO = re.compile(r"pts_time:\s*([0-9.]+).*?mean:\[\s*([0-9.]+)")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=False, check=False)


def probe(path: Path) -> dict:
    """Duration, nominal fps, frame count, resolution, audio, creation time, device model."""
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
              "-show_streams", str(path)])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.decode(errors='replace')[-300:]}")
    info = json.loads(r.stdout)
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError(f"{path} has no video stream")
    vtags = {k.lower(): v for k, v in (video.get("tags") or {}).items()}
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            f = Fraction(video.get(key, "0/1"))
            if f > 0:
                fps = float(f)
                break
        except (ValueError, ZeroDivisionError):
            pass
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None
    rotate = vtags.get("rotate")
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotate = sd["rotation"]
    return {
        "duration_s": round(duration, 3),
        "fps": round(fps, 3) if fps else None,
        "frames": frames,
        "width": video.get("width"),
        "height": video.get("height"),
        "rotation": rotate,
        "codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "creation_time": tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
                         or vtags.get("creation_time"),
        "make": tags.get("com.apple.quicktime.make"),
        "model": tags.get("com.apple.quicktime.model"),
        "software": tags.get("com.apple.quicktime.software") or tags.get("encoder"),
    }


def luminance(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (pts seconds, mean luma 0-255), using real frame timestamps (VFR-safe)."""
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-an",
              "-vf", "scale=96:-2,showinfo", "-f", "null", "-"])
    t, y = [], []
    for line in r.stderr.decode(errors="replace").splitlines():
        if "showinfo" in line and "pts_time" in line:
            m = _SHOWINFO.search(line)
            if m:
                t.append(float(m.group(1)))
                y.append(float(m.group(2)))
    if not t:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.asarray(t), np.asarray(y)


def audio(path: Path, rate: int = 8000) -> np.ndarray | None:
    """Mono PCM float32 in [-1, 1] at `rate` Hz, or None if the clip has no audio."""
    r = _run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-i", str(path),
              "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"])
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype="<i2").astype(np.float32) / 32768.0


def sample_frames(path: Path, times: list[float], width: int = 640) -> list[np.ndarray]:
    """RGB frames at the given times (for optional hand detection)."""
    out = []
    for t in times:
        r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i",
                  str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "rawvideo",
                  "-pix_fmt", "rgb24", "-"])
        if r.returncode != 0 or not r.stdout:
            continue
        h = len(r.stdout) // (width * 3)
        if h:
            out.append(np.frombuffer(r.stdout[: h * width * 3], np.uint8).reshape(h, width, 3))
    return out
