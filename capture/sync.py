"""Find the shared sync events in each stream.

Primary: the light flick. Turning the room light off makes every camera's mean
brightness fall sharply within one or two frames. The RealSense has no microphone,
so a visual event is the only one all three streams share.

Secondary: the clap, found in the phone and MacBook audio. It cross-checks the
light-event alignment between those two streams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_DROP = 12.0      # luma levels (0-255) the light-off edge must fall
MIN_Z = 6.0          # how far the edge stands out from ordinary frame-to-frame noise


@dataclass
class Event:
    t: float          # seconds from the start of the stream
    strength: float   # z-score of the edge (or clap peak over the noise floor)
    drop: float = 0.0


def light_off_event(t: np.ndarray, y: np.ndarray, window_s: float = 0.25,
                    search_s: float | None = None) -> Event | None:
    """Time of the sharpest brightness fall (light off).

    Compares mean brightness just before and just after each frame over `window_s`,
    so one dark frame from auto-exposure hunting does not count, and a slow fade does
    not either. `search_s` limits the search to the first N seconds of the clip.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 8:
        return None
    if search_s is not None:
        keep = t <= t[0] + search_s
        t, y = t[keep], y[keep]
        if len(t) < 8:
            return None
    dt = np.median(np.diff(t)) or 1 / 30
    k = max(1, int(round(window_s / dt)))
    idx = np.arange(k, len(y) - k + 1)
    if len(idx) == 0:
        return None
    # Medians, so one dark frame (auto-exposure hunting, a hand over the lens for a
    # frame) cannot pass for the light going off; the light must stay off ~window_s.
    win = np.median(np.lib.stride_tricks.sliding_window_view(y, k), axis=1)  # win[j] = y[j:j+k]
    before = win[idx - k]
    after = win[idx]
    step = after - before                      # negative at light-off
    i = int(np.argmin(step))
    drop = -float(step[i])
    noise = np.diff(y)
    mask = np.ones(len(noise), bool)
    mask[max(0, idx[i] - 2 * k): idx[i] + 2 * k] = False
    sigma = float(np.median(np.abs(noise[mask] - np.median(noise[mask])))) * 1.4826 if mask.any() else 0.0
    sigma = max(sigma, 0.5)
    # Edge frame: the first frame after `before` that is clearly darker.
    j0 = idx[i] - k
    seg = y[j0: idx[i] + k]
    thresh = before[i] - 0.5 * drop
    below = np.nonzero(seg < thresh)[0]
    edge = j0 + int(below[0]) if len(below) else int(idx[i])
    z = drop / sigma
    if drop < MIN_DROP or z < MIN_Z:
        return None
    return Event(t=float(t[edge] - t[0]), strength=round(z, 1), drop=round(drop, 1))


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
