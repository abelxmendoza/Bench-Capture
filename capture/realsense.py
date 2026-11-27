"""RealSense D455: record a .bag, and summarize one back without a camera attached.

pyrealsense2 is imported lazily so the rest of the pipeline (and its tests) runs on
machines without librealsense.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

DEFAULT = {"width": 848, "height": 480, "fps": 30}


def _rs():
    import pyrealsense2 as rs  # noqa: PLC0415
    return rs


def devices() -> list[dict]:
    rs = _rs()
    out = []
    for d in rs.context().query_devices():
        info = {}
        for key, name in ((rs.camera_info.name, "name"), (rs.camera_info.serial_number, "serial"),
                          (rs.camera_info.firmware_version, "firmware"),
                          (rs.camera_info.usb_type_descriptor, "usb")):
            try:
                info[name] = d.get_info(key)
            except RuntimeError:
                pass
        out.append(info)
    return out


class Recorder:
    """Background recorder: start(), then stop() from any thread."""

    def __init__(self, bag: Path, width=DEFAULT["width"], height=DEFAULT["height"],
                 fps=DEFAULT["fps"], serial: str | None = None):
        self.bag, self.width, self.height, self.fps, self.serial = bag, width, height, fps, serial
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.frames = 0
        self.started_host_s: float | None = None
        self.device: dict = {}

    def start(self, warmup_s: float = 5.0) -> None:
        rs = _rs()
        found = devices()
        if not found:
            raise RuntimeError("no RealSense camera found - is it plugged into a USB 3 port?")
        self.device = next((d for d in found if d.get("serial") == self.serial), found[0])
        if not str(self.device.get("usb", "3")).startswith("3"):
            raise RuntimeError(f"RealSense is on USB {self.device.get('usb')}; use a USB 3 port "
                               "or frames will drop at 848x480@30")
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg.enable_record_to_file(str(self.bag))
        self._pipe.start(cfg)
        self.started_host_s = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        deadline = time.time() + warmup_s
        while self.frames == 0 and time.time() < deadline and self.error is None:
            time.sleep(0.05)
        if self.error:
            raise RuntimeError(f"RealSense failed to start: {self.error}")
        if self.frames == 0:
            self.stop()
            raise RuntimeError("RealSense started but produced no frames")

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, _ = self._pipe.try_wait_for_frames(1000)
                if ok:
                    self.frames += 1
        except BaseException as e:  # noqa: BLE001 - reported to the caller
            self.error = e

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self._pipe.stop()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class Frame:
    stream: str                 # "depth" | "color"
    number: int
    t_ms: float                 # device/global timestamp
    luma: float | None = None   # color only: mean brightness 0-255
    valid: float | None = None  # depth only: fraction of non-zero pixels


@dataclass
class BagSummary:
    streams: dict = field(default_factory=dict)
    color_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    color_luma: np.ndarray = field(default_factory=lambda: np.zeros(0))


def read_bag(bag: Path, depth_every: int = 10) -> Iterator[Frame]:
    """Every frame in the bag, as fast as it decodes (not real time)."""
    rs = _rs()
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(str(bag), repeat_playback=False)
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    n = 0
    try:
        while True:
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            for f in fs:
                st = f.get_profile().stream_type()
                if st == rs.stream.color:
                    a = np.asanyarray(f.get_data())[::8, ::8]
                    yield Frame("color", f.get_frame_number(), f.get_timestamp(),
                                luma=float(a.mean()))
                elif st == rs.stream.depth:
                    n += 1
                    valid = None
                    if n % depth_every == 1:
                        valid = float((np.asanyarray(f.get_data())[::4, ::4] > 0).mean())
                    yield Frame("depth", f.get_frame_number(), f.get_timestamp(), valid=valid)
    finally:
        pipe.stop()


def summarize(frames: Iterable[Frame], nominal_fps: float = DEFAULT["fps"]) -> BagSummary:
    """Per-stream frame counts, duration, dropped frames and depth coverage."""
    by: dict[str, dict[int, Frame]] = {}
    for f in frames:
        by.setdefault(f.stream, {})[f.number] = f      # playback can repeat a frame; dedupe
    out = BagSummary()
    for name, fmap in by.items():
        nums = sorted(fmap)
        ts = np.array([fmap[k].t_ms for k in nums]) / 1000.0
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        expected = int(round(dur * nominal_fps)) + 1 if dur else len(nums)
        gaps = np.diff(ts) if len(ts) > 1 else np.zeros(0)
        dropped = int(np.sum(np.maximum(np.round(gaps * nominal_fps) - 1, 0)))
        info = {"frames": len(nums), "duration_s": round(dur, 3), "expected_frames": expected,
                "dropped_frames": dropped,
                "drop_rate": round(dropped / max(expected, 1), 4)}
        if name == "depth":
            v = [fmap[k].valid for k in nums if fmap[k].valid is not None]
            info["valid_pixel_fraction_median"] = round(float(np.median(v)), 3) if v else 0.0
        out.streams[name] = info
        if name == "color":
            out.color_t = ts - ts[0]
            out.color_luma = np.array([fmap[k].luma for k in nums], float)
    return out
