"""Live RealSense brightness readout for setting up lights before recording.

Prints twice a second: brightness, a bar, depth coverage, and FLICK when an edge
that would pass the sync check just happened. Nothing is recorded.
"""
from __future__ import annotations

import time

import numpy as np

from . import sync

BAR = 30


def _bar(v: float) -> str:
    n = int(round(v / 255 * BAR))
    return "#" * n + "." * (BAR - n)


def save_snapshot(path, color: np.ndarray, depth: np.ndarray) -> bool:
    """Color frame beside a colorized depth map, as one PNG (needs OpenCV)."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return False
    dep = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.06), cv2.COLORMAP_JET)
    return bool(cv2.imwrite(str(path), np.hstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), dep])))


def run(seconds: float = 30.0, width: int = 848, height: int = 480, fps: int = 30,
        serial: str | None = None, out=print, snapshot=None) -> dict:
    import pyrealsense2 as rs  # noqa: PLC0415
    pipe, cfg = rs.pipeline(), rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipe.start(cfg)
    t, y = [], []
    flicks: list[sync.Event] = []
    depth_valid = 0.0
    last = None
    start = time.time()
    next_print = start
    try:
        while time.time() - start < seconds:
            ok, fs = pipe.try_wait_for_frames(1000)
            if not ok:
                continue
            now = time.time() - start
            c = fs.get_color_frame()
            if c:
                t.append(now)
                y.append(float(np.asanyarray(c.get_data())[::8, ::8].mean()))
            d = fs.get_depth_frame()
            if d and len(t) % 10 == 1:
                depth_valid = float((np.asanyarray(d.get_data())[::4, ::4] > 0).mean())
            if snapshot and c and d and len(t) % 15 == 0:
                last = (np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy())
            if time.time() >= next_print and t:
                next_print += 0.5
                mark = ""
                # look for a flick in the last ~3 s that we have not reported yet
                w0 = max(0, len(t) - 3 * fps)
                ev = None
                if now > sync.SKIP_S:      # auto-exposure is still settling before this
                    t_w, y_w = np.array(t[w0:]), np.array(y[w0:])
                    keep = t_w >= sync.SKIP_S
                    ev = sync.light_event(t_w[keep], y_w[keep], skip_s=0) if keep.sum() > 8 else None
                    if ev is not None:
                        w0 = int(np.nonzero(np.array(t) >= t_w[keep][0])[0][0])
                if ev is not None:
                    abs_t = t[w0] + ev.t
                    if not flicks or abs_t - (t[0] + flicks[-1].t) > 1.0:
                        flicks.append(sync.Event(abs_t - t[0], ev.strength, ev.drop, ev.direction, ev.rel))
                        mark = f"  << FLICK {ev.direction} ({ev.rel:.0%}, z={ev.strength})"
                lt = sync.lighting(y[-fps:])
                out(f"{now:6.1f}s  brightness {y[-1]:5.1f} |{_bar(y[-1])}|  depth {depth_valid:4.0%}"
                    f"  [{lt['verdict']}]{mark}")
    finally:
        pipe.stop()
    lt = sync.lighting(y)
    whole = sync.light_event(np.array(t), np.array(y)) if len(t) > 8 else None
    summary = {"frames": len(y), "lighting": lt, "flicks_seen": len(flicks),
               "first_flick": None if whole is None else
               {"t": round(whole.t, 2), "direction": whole.direction, "rel": whole.rel, "z": whole.strength}}
    out("")
    if snapshot and last is not None:
        if save_snapshot(snapshot, *last):
            summary["snapshot"] = str(snapshot)
            out(f"snapshot (color | depth): {snapshot}")
    out(f"brightness median {lt.get('median', 0):.0f}/255 (p5 {lt.get('p5', 0):.0f}, p95 {lt.get('p95', 0):.0f})"
        f" -> {lt.get('verdict', '?')}")
    if whole:
        out(f"sync check WOULD PASS: first flick at {whole.t:.1f}s ({whole.direction}, "
            f"{whole.rel:.0%} change, z={whole.strength})")
    else:
        out("sync check would FAIL: no flick seen. Make the light change bigger "
            f"(at least {sync.MIN_REL:.0%} brighter/darker and held ~1 s) or aim the camera at what the light hits.")
    return summary
