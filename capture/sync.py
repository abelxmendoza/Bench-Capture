"""Find the shared sync events in each stream.

Primary: the light flick. Switching a light off (or a flashlight on, in a dark room)
changes every camera's mean brightness within a frame or two. The RealSense has no
microphone, so a visual event is the only one all three streams share.

The first qualifying brightness edge in a clip is the sync point, in either
direction. With "lamp off, then back on", every stream locks to the off edge; with
"flashlight on, then off" every stream locks to the on edge. Taking the first edge
rather than the biggest keeps all streams on the same physical event.

The threshold scales with the scene: an edge counts if it changes brightness by at
least MIN_REL of the brighter level (a small lamp in a dark room still counts), by at
least MIN_ABS levels (sensor noise in near-black scenes does not), and stands well
out from ordinary frame-to-frame noise.

Secondary: the clap, found in the phone and MacBook audio, cross-checks alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_ABS = 6.0        # luma levels (0-255)
MIN_REL = 0.20       # fraction of the brighter side of the edge
MIN_Z = 6.0          # edge size over the clip's frame-to-frame noise (robust sigma)
SKIP_S = 1.5         # ignore the start of every clip: camera auto-exposure ramps up there
DARK = 40.0          # median luma below this: too dark for usable color / hand detection
BRIGHT = 225.0       # median luma above this: blown out


@dataclass
class Event:
    t: float               # seconds from the start of the stream
    strength: float        # z-score of the edge (or clap peak over the noise floor)
    drop: float = 0.0      # absolute brightness change, luma levels
    direction: str = ""    # "off" (darker) or "on" (brighter); "" for claps
    rel: float = 0.0       # change as a fraction of the brighter level


def _steps(y: np.ndarray, k: int):
    """Median of the k frames after minus median of the k frames before, per frame."""
    win = np.median(np.lib.stride_tricks.sliding_window_view(y, k), axis=1)  # win[j] = med(y[j:j+k])
    idx = np.arange(k, len(y) - k + 1)
    return idx, win[idx - k], win[idx]


def _sigma(y: np.ndarray, exclude: slice) -> float:
    noise = np.diff(y)
    mask = np.ones(len(noise), bool)
    mask[exclude] = False
    if not mask.any():
        return 0.5
    n = noise[mask]
    return max(float(np.median(np.abs(n - np.median(n)))) * 1.4826, 0.5)


def light_event(t: np.ndarray, y: np.ndarray, window_s: float = 0.25,
                search_s: float | None = None, skip_s: float = SKIP_S) -> Event | None:
    """The first light flick (either direction) in the clip, or None.

    Brightness must stay changed for about `window_s`, so one dark frame from
    auto-exposure hunting or a hand passing the lens does not count, and a slow fade
    does not either. `search_s` limits the search to the first N seconds; the first
    `skip_s` seconds are ignored because every camera's auto-exposure ramps up there.
    Returned times are still measured from the start of the clip.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    t_start = float(t[0]) if len(t) else 0.0
    if skip_s and len(t):
        keep = t >= t_start + skip_s
        t, y = t[keep], y[keep]
    if search_s is not None and len(t):
        keep = t <= t_start + search_s
        t, y = t[keep], y[keep]
    if len(t) < 8:
        return None
    dt = float(np.median(np.diff(t))) or 1 / 30
    k = max(2, int(round(window_s / dt)))
    idx, before, after = _steps(y, k)
    if len(idx) == 0:
        return None
    step = after - before
    mag = np.abs(step)
    brighter = np.maximum(np.maximum(before, after), 1.0)
    rel = mag / brighter
    # Noise from the whole clip, minus the biggest edge, so a flick does not inflate it.
    big = int(idx[np.argmax(mag)])
    sigma = _sigma(y, slice(max(0, big - 2 * k), big + 2 * k))
    ok = (mag >= MIN_ABS) & (rel >= MIN_REL) & (mag / sigma >= MIN_Z)
    hits = np.nonzero(ok)[0]
    if len(hits) == 0:
        return None
    first = hits[0]
    # Peak of that edge: the largest change within the next 2k frames of the same sign.
    sign = np.sign(step[first])
    span = np.arange(first, min(len(step), first + 2 * k))
    span = span[np.sign(step[span]) == sign]
    i = int(span[np.argmax(mag[span])])
    # Edge frame: first frame past the halfway point between the before/after levels.
    j0 = int(idx[i] - k)
    seg = y[j0: idx[i] + k]
    half = before[i] + 0.5 * step[i]
    crossed = np.nonzero(seg < half)[0] if sign < 0 else np.nonzero(seg > half)[0]
    edge = j0 + int(crossed[0]) if len(crossed) else int(idx[i])
    return Event(t=float(t[edge] - t_start), strength=round(float(mag[i] / sigma), 1),
                 drop=round(float(mag[i]), 1), direction="off" if sign < 0 else "on",
                 rel=round(float(rel[i]), 3))


light_off_event = light_event   # backward-compatible name


def lighting(y: np.ndarray) -> dict:
    """Brightness summary of a clip, for the lighting check and the manifest."""
    y = np.asarray(y, float)
    if len(y) == 0:
        return {}
    med = float(np.median(y))
    verdict = "too dark" if med < DARK else "blown out" if med > BRIGHT else "ok"
    return {"median": round(med, 1), "p5": round(float(np.percentile(y, 5)), 1),
            "p95": round(float(np.percentile(y, 95)), 1), "verdict": verdict}


def clap_event(pcm: np.ndarray | None, rate: int, after_s: float = 0.0,
               within_s: float = 15.0) -> Event | None:
    """Loudest short transient in [after_s, after_s + within_s]."""
    if pcm is None or len(pcm) < rate // 10:
        return None
    a0 = int(after_s * rate)
    a1 = min(len(pcm), int((after_s + within_s) * rate))
    seg = np.abs(pcm[a0:a1])
    if len(seg) < rate // 20:
        return None
    hop = max(1, rate // 1000)                 # 1 ms envelope
    n = len(seg) // hop
    env = seg[: n * hop].reshape(n, hop).max(axis=1)
    peak = int(np.argmax(env))
    floor = float(np.median(env)) + 1e-4
    z = float(env[peak]) / floor
    if env[peak] < 0.05 or z < 8.0:
        return None
    onset = peak
    while onset > 0 and env[onset - 1] > 0.3 * env[peak]:
        onset -= 1
    return Event(t=after_s + onset * hop / rate, strength=round(z, 1))
