"""Capture pipeline tests. Synthetic clips are built with ffmpeg; no camera needed."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from capture import cli, hivemind, ingest, media, sync, validate  # noqa: E402
from capture.realsense import Frame, summarize  # noqa: E402
from capture.session import Session  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
RATE = 8000


# ---------------------------------------------------------------- helpers

def luma_series(fps=30, dur=6.0, flick=2.0, noise=1.5, seed=0, bright=150, dark=40, off_s=0.6):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fps)
    y = np.full_like(t, bright, dtype=float) + rng.normal(0, noise, len(t))
    if flick is not None:
        y[(t >= flick) & (t < flick + off_s)] = dark
    return t, y


def make_clip(path: Path, *, fps=30, dur=6.0, flick=2.0, clap=3.0, start: datetime | None = None,
              model: str | None = None, size=(160, 120), audio=True):
    """A gray clip whose brightness drops at `flick` for 0.6 s, with a click at `clap`."""
    w, h = size
    n = int(round(dur * fps))
    frames = np.full((n, h, w), 150, np.uint8)
    if flick is not None:
        a, b = int(round(flick * fps)), int(round((flick + 0.6) * fps))
        frames[a:b] = 40
    yuv = np.concatenate([frames, np.full((n, h // 2, w), 128, np.uint8)], axis=1)  # gray -> fake I420 planes
    raw = path.with_suffix(".yuv")
    raw.write_bytes(yuv.reshape(n, -1).tobytes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{int(h * 1.5)}", "-r", str(fps), "-i", str(raw)]
    if audio:
        pcm = np.zeros(int(dur * RATE), np.float32)
        if clap is not None:
            i = int(clap * RATE)
            pcm[i:i + 40] = np.hanning(40) * 0.9
        wav = path.with_suffix(".s16")
        wav.write_bytes((pcm * 32767).astype("<i2").tobytes())
        cmd += ["-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", str(wav)]
    cmd += ["-vf", f"crop={w}:{h}:0:0,format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
            "-qp", "0", "-g", "1"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "96k"]
    if start:
        cmd += ["-metadata", "creation_time=" + start.strftime("%Y-%m-%dT%H:%M:%S.000000Z")]
    if model:
        cmd += ["-metadata", f"com.apple.quicktime.model={model}", "-movflags", "use_metadata_tags"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    raw.unlink()
    if audio:
        path.with_suffix(".s16").unlink()
    return path


# ---------------------------------------------------------------- sync

def test_light_off_found_to_one_frame():
    t, y = luma_series(flick=2.0)
    ev = sync.light_off_event(t, y)
    assert ev is not None and abs(ev.t - 2.0) <= 1 / 30 + 1e-9
    assert ev.drop > 50 and ev.strength >= sync.MIN_Z


def test_single_dark_frame_is_not_a_flick():
    t, y = luma_series(flick=None)
    y[60] = 40          # one auto-exposure glitch
    assert sync.light_off_event(t, y) is None


def test_slow_fade_is_not_a_flick():
    t = np.arange(0, 10, 1 / 30)
    y = 150 - 10 * t    # sunset, 100 levels over 10 s
    assert sync.light_off_event(t, y) is None


def test_flick_search_window():
    t, y = luma_series(dur=20, flick=15)
    assert sync.light_off_event(t, y, search_s=10) is None
    assert sync.light_off_event(t, y, search_s=None) is not None


def test_clap_onset_and_silence():
    pcm = np.random.default_rng(1).normal(0, 0.002, RATE * 5).astype(np.float32)
    pcm[int(2.5 * RATE): int(2.5 * RATE) + 50] = 0.8
    ev = sync.clap_event(pcm, RATE)
    assert ev is not None and abs(ev.t - 2.5) < 0.005
    assert sync.clap_event(np.zeros(RATE * 5, np.float32), RATE) is None
    # clap before the flick is ignored when searching after it
    assert sync.clap_event(pcm, RATE, after_s=3.0) is None


# ---------------------------------------------------------------- realsense summary

def test_bag_summary_counts_drops_and_dedupes():
    frames = []
    for i in range(300):
        if i in (100, 101, 102):        # 3 dropped frames
            continue
        frames.append(Frame("depth", i, i * 1000 / 30, valid=0.8 if i % 10 == 1 else None))
        frames.append(Frame("color", i, i * 1000 / 30, luma=150.0))
    frames.append(Frame("color", 5, 5 * 1000 / 30, luma=150.0))   # repeated by playback
    s = summarize(frames)
    assert s.streams["depth"]["dropped_frames"] == 3
    assert s.streams["color"]["frames"] == 297
    assert s.streams["depth"]["valid_pixel_fraction_median"] == pytest.approx(0.8)
    assert len(s.color_t) == 297 and s.color_t[0] == 0


# ---------------------------------------------------------------- media + ingest

@needs_ffmpeg
def test_media_probe_luma_audio(tmp_path):
    f = make_clip(tmp_path / "c.mov", model="iPhone 15")
    info = media.probe(f)
    assert info["fps"] == 30 and info["has_audio"] and info["model"] == "iPhone 15"
    t, y = media.luminance(f)
    assert len(t) == 180
    ev = sync.light_off_event(t, y)
    assert abs(ev.t - 2.0) <= 1 / 30 + 1e-6
    clap = sync.clap_event(media.audio(f, RATE), RATE, after_s=ev.t)
    assert clap is not None and abs(clap.t - 3.0) < 0.06     # aac priming delay is small


def test_classify():
    assert ingest.classify(Path("IMG_0001.MOV"), {"model": "iPhone 15", "make": "Apple"}) == "phone"
    assert ingest.classify(Path("Movie Recording.mov"), {"model": None}) == "mac"
    assert ingest.classify(Path("take3_mac.mov"), {"model": "iPhone 15"}) == "mac"
    assert ingest.classify(Path("x.mp4"), {"model": "GoPro HERO11"}) is None


def test_match_take_start_or_end_stamp():
    t0 = datetime(2026, 11, 26, 20, 0, tzinfo=timezone.utc)
    takes = [{"take": 1, "start_utc": t0.isoformat(), "end_utc": (t0 + timedelta(seconds=60)).isoformat()},
             {"take": 2, "start_utc": (t0 + timedelta(minutes=10)).isoformat(),
              "end_utc": (t0 + timedelta(minutes=11)).isoformat()}]
    start_stamped = {"creation_time": (t0 + timedelta(minutes=10, seconds=-5)).isoformat(), "duration_s": 70}
    end_stamped = {"creation_time": (t0 + timedelta(seconds=65)).isoformat(), "duration_s": 70}
    assert ingest.match_take(start_stamped, takes) == 2
    assert ingest.match_take(end_stamped, takes) == 1
    assert ingest.match_take({"creation_time": (t0 + timedelta(hours=3)).isoformat(), "duration_s": 5}, takes) is None
    assert ingest.match_take({"duration_s": 5}, takes) is None


def _session_with_take(tmp_path, devices=("realsense", "phone", "mac")):
    s = Session.open("bench1", root=tmp_path, task="pick and place", devices=devices)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    s.record_take({"take": 1, "start_utc": start.isoformat(),
                   "end_utc": (start + timedelta(seconds=6)).isoformat(), "devices": list(devices)})
    return s, start


@needs_ffmpeg
def test_ingest_files_clips_under_take(tmp_path):
    s, start = _session_with_take(tmp_path)
    make_clip(s.path / "inbox" / "IMG_4410.MOV", start=start - timedelta(seconds=3), model="iPhone 15")
    make_clip(s.path / "inbox" / "Movie Recording.mov", start=start - timedelta(seconds=2))
    make_clip(s.path / "inbox" / "old.mov", start=start - timedelta(days=2))
    res = {r["file"]: r for r in ingest.ingest(s, taildrop=False)}
    assert res["IMG_4410.MOV"]["dest"] == "take_01/phone.mov"
    assert res["Movie Recording.mov"]["dest"] == "take_01/mac.mov"
    assert res["old.mov"]["status"].startswith("unmatched")
    assert (s.path / "inbox" / "old.mov").exists()


# ---------------------------------------------------------------- validate

def fake_bag(flick=2.0, depth_valid=0.7, drops=0, dur=6.0):
    def analyze(path):
        t, y = luma_series(dur=dur, flick=flick, seed=3)
        n = len(t)
        return {"streams": {"depth": {"frames": n - drops, "duration_s": dur, "expected_frames": n,
                                      "dropped_frames": drops, "drop_rate": drops / n,
                                      "valid_pixel_fraction_median": depth_valid},
                            "color": {"frames": n, "duration_s": dur, "expected_frames": n,
                                      "dropped_frames": 0, "drop_rate": 0.0}},
                "duration_s": dur, "light_event": sync.light_off_event(t, y)}
    return analyze


def hands_pass(path, start, end):
    return validate._check("hands_in_frame", "PASS", "test stub")


def _filled_take(tmp_path, mac_flick=2.0, mac_clap=3.0):
    s, start = _session_with_take(tmp_path)
    d = s.take_dir(1)
    (d / "realsense.bag").write_bytes(b"stub")
    make_clip(d / "phone.mov", flick=2.0, clap=3.0, model="iPhone 15")
    make_clip(d / "mac.mov", flick=mac_flick, clap=mac_clap)
    return s


@needs_ffmpeg
def test_validate_take_pass_and_manifest(tmp_path):
    s = _filled_take(tmp_path)
    m = validate.validate_take(s, 1, {"bag": fake_bag(), "hands": hands_pass})
    status = {c["check"]: c["status"] for c in m["checks"]}
    assert m["verdict"] == "PASS", m["checks"]
    assert status["sync_light_flick"] == status["clap_crosscheck"] == status["depth_present"] == "PASS"
    on_disk = json.loads((s.path / "take_01" / "manifest.json").read_text())
    assert on_disk["streams"]["phone"]["sync_offset_s"] == pytest.approx(2.0, abs=0.04)
    assert len(on_disk["streams"]["mac"]["sha256"]) == 64


@needs_ffmpeg
def test_validate_fails_without_flick_and_with_bad_depth(tmp_path):
    s = _filled_take(tmp_path, mac_flick=None, mac_clap=None)
    m = validate.validate_take(s, 1, {"bag": fake_bag(depth_valid=0.05, drops=20), "hands": hands_pass})
    status = {c["check"]: c["status"] for c in m["checks"]}
    assert m["verdict"] == "FAIL"
    assert status["sync_light_flick"] == "FAIL"
    assert status["depth_present"] == "FAIL"
    assert status["realsense_depth_drops"] == "FAIL"
    assert status["clap_crosscheck"] == "WARN"


@needs_ffmpeg
def test_clap_disagreement_warns(tmp_path):
    s = _filled_take(tmp_path, mac_clap=3.4)      # Mac clip misaligned by 400 ms
    m = validate.validate_take(s, 1, {"bag": fake_bag(), "hands": hands_pass})
    c = next(c for c in m["checks"] if c["check"] == "clap_crosscheck")
    assert c["status"] == "WARN" and c["error_ms"] > 300


def test_missing_files_fail(tmp_path):
    s, _ = _session_with_take(tmp_path)
    m = validate.validate_take(s, 1)
    assert m["verdict"] == "FAIL"
    assert m["checks"][0]["check"] == "files_present" and m["checks"][0]["status"] == "FAIL"


# ---------------------------------------------------------------- hivemind + cli

def test_hivemind_log_replaces_row(tmp_path):
    hm = tmp_path / "agent-hivemind"
    hm.mkdir()
    summary = {"session": "bench1", "task": "t", "takes": [{}, {}], "passed": 1, "failed": 1}
    hivemind.log_session(summary, tmp_path / "bench1", hm)
    summary["passed"], summary["failed"] = 2, 0
    hivemind.log_session(summary, tmp_path / "bench1", hm)
    rows = list(csv.DictReader((hm / "state" / "capture_sessions.csv").open()))
    assert len(rows) == 1 and rows[0]["passed"] == "2"
    assert "skipped" in hivemind.log_session(summary, tmp_path, tmp_path / "nope")


def test_cli_take_without_realsense(tmp_path, capsys):
    rc = cli.main(["--root", str(tmp_path), "take", "dry", "--devices", "phone,mac",
                   "--yes", "--duration", "0", "--task", "dry run"])
    assert rc == 0
    s = Session(tmp_path / "dry")
    assert s.takes[0]["take"] == 1 and s.meta["task"] == "dry run"
    assert "Taildrop" in capsys.readouterr().out


def test_cli_rejects_unknown_device(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--root", str(tmp_path), "take", "x", "--devices", "gopro", "--yes", "--duration", "0"])


# ---------------------------------------------------------------- lighting variations

def _series(levels, fps=30, noise=0.6, seed=7):
    """Piecewise-constant brightness: levels = [(seconds, luma), ...]."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(int(s * fps), v, float) for s, v in levels])
    y += rng.normal(0, noise, len(y))
    return np.arange(len(y)) / fps, y


def test_dark_room_small_lamp_counts():
    # dim room (46) with a small desk lamp switched off for 1 s -> 34 (26% darker)
    t, y = _series([(3, 46), (1, 34), (4, 46)])
    ev = sync.light_event(t, y)
    assert ev is not None and ev.direction == "off" and abs(ev.t - 3.0) <= 1 / 30 + 1e-9
    assert ev.rel == pytest.approx(0.26, abs=0.03)


def test_flashlight_on_in_dark_room_counts():
    t, y = _series([(3, 12), (1, 60), (4, 12)])
    ev = sync.light_event(t, y)
    assert ev is not None and ev.direction == "on" and abs(ev.t - 3.0) <= 1 / 30 + 1e-9


def test_first_edge_wins_off_then_on():
    # off edge at 2 s is smaller than the on edge at 3 s; all streams must lock to the first one
    t, y = _series([(2, 120), (1, 80), (3, 200)])
    ev = sync.light_event(t, y)
    assert ev.direction == "off" and abs(ev.t - 2.0) <= 1 / 30 + 1e-9


def test_near_black_noise_is_not_a_flick():
    # 4 -> 2 is a 50% change but only 2 levels: sensor noise territory
    t, y = _series([(3, 4), (1, 2), (3, 4)], noise=0.3)
    assert sync.light_event(t, y) is None


def test_small_relative_change_in_bright_room_is_not_a_flick():
    # someone walks past: 200 -> 185 (7.5%)
    t, y = _series([(3, 200), (1, 185), (3, 200)])
    assert sync.light_event(t, y) is None


def test_lighting_summary():
    assert sync.lighting(np.full(100, 20.0))["verdict"] == "too dark"
    assert sync.lighting(np.full(100, 240.0))["verdict"] == "blown out"
    assert sync.lighting(np.full(100, 120.0))["verdict"] == "ok"


@needs_ffmpeg
def test_validate_reports_lighting_and_direction(tmp_path):
    s = _filled_take(tmp_path)
    m = validate.validate_take(s, 1, {"bag": fake_bag(), "hands": hands_pass})
    names = {c["check"]: c for c in m["checks"]}
    assert names["phone_lighting"]["status"] == "PASS"
    assert "sync_same_edge" not in names
    assert m["streams"]["phone"]["light_event"]["direction"] == "off"
    assert m["streams"]["phone"]["lighting"]["median"] == pytest.approx(150, abs=8)   # codec range shifts it a little


@needs_ffmpeg
def test_mismatched_edges_fail(tmp_path):
    s = _filled_take(tmp_path)

    def bag_on_edge(path):
        t, y = _series([(2, 30), (1, 90), (3, 30)])
        return {"streams": {"depth": {"frames": len(t), "duration_s": 6.0, "expected_frames": len(t),
                                      "dropped_frames": 0, "drop_rate": 0.0,
                                      "valid_pixel_fraction_median": 0.7}},
                "duration_s": 6.0, "light_event": sync.light_event(t, y), "lighting": sync.lighting(y)}

    m = validate.validate_take(s, 1, {"bag": bag_on_edge, "hands": hands_pass})
    names = {c["check"]: c["status"] for c in m["checks"]}
    assert names["sync_same_edge"] == "FAIL" and m["verdict"] == "FAIL"
    assert names["realsense_lighting"] == "WARN"      # median 30: too dark


def test_auto_exposure_warmup_is_not_the_flick():
    # what the D455 really does at start (seen on the Orin): 2 -> 30 -> 45 within 0.6 s,
    # then steady; the real flick comes at 4 s and must be the one reported
    t, y = _series([(0.3, 2), (0.3, 30), (3.4, 45), (1, 25), (3, 45)])
    ev = sync.light_event(t, y)
    assert ev is not None and ev.direction == "off" and abs(ev.t - 4.0) <= 1 / 30 + 1e-9
    # and with no flick at all, the warmup alone must not pass the sync check
    t, y = _series([(0.3, 2), (0.3, 30), (6, 45)])
    assert sync.light_event(t, y) is None
