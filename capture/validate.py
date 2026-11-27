"""Per-take checks -> manifest.json with a PASS / FAIL verdict.

FAIL means the take is not usable as training data as recorded (redo it).
WARN means usable, but something is worth a look. SKIP means a check could not run.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from . import __version__, media, sync
from .session import Session, utc_now

SYNC_TOL_S = 0.1            # phone-vs-Mac: light-flick vs clap alignment must agree within this
MAX_DROP_RATE = 0.01
MIN_DEPTH_VALID = 0.30
MIN_FPS_RATIO = 0.90        # phone/Mac effective fps vs nominal
DURATION_TOL_S = 2.0        # streams' time after the light flick must agree within this
MIN_HAND_FRACTION = 0.6
SEARCH_S = 60.0             # light flick must happen in the first minute of every clip
RATE = 8000


def _sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str | None:
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def _check(name, status, detail, **data):
    return {"check": name, "status": status, "detail": detail, **data}


def analyze_video(path: Path) -> dict:
    info = media.probe(path)
    t, y = media.luminance(path)
    ev = sync.light_event(t, y, search_s=SEARCH_S)
    pcm = media.audio(path, RATE) if info["has_audio"] else None
    clap = sync.clap_event(pcm, RATE, after_s=ev.t if ev else 0.0) if pcm is not None else None
    eff_fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else None
    return {"info": info, "decoded_frames": len(t), "effective_fps": round(eff_fps, 2) if eff_fps else None,
            "duration_s": round(float(t[-1] - t[0]), 3), "light_event": ev, "clap_event": clap,
            "lighting": sync.lighting(y)}


def analyze_bag(path: Path) -> dict:
    from . import realsense  # noqa: PLC0415 - needs pyrealsense2
    s = realsense.summarize(realsense.read_bag(path))
    ev = None
    if len(s.color_t):
        ev = sync.light_event(s.color_t, s.color_luma, search_s=SEARCH_S)
    dur = s.streams.get("color", s.streams.get("depth", {})).get("duration_s", 0.0)
    return {"streams": s.streams, "duration_s": dur, "light_event": ev,
            "lighting": sync.lighting(s.color_luma)}


def hands_check(path: Path, start: float, end: float, n: int = 12):
    try:
        import mediapipe as mp  # noqa: PLC0415
    except ImportError:
        return _check("hands_in_frame", "SKIP", "mediapipe not installed in this environment")
    times = list(np.linspace(start + 0.5, max(start + 0.6, end - 0.5), n))
    frames = media.sample_frames(path, times)
    if not frames:
        return _check("hands_in_frame", "SKIP", "could not decode sample frames")
    hits = 0
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=0.5) as hands:
        for fr in frames:
            if hands.process(fr).multi_hand_landmarks:
                hits += 1
    frac = hits / len(frames)
    status = "PASS" if frac >= MIN_HAND_FRACTION else "FAIL"
    return _check("hands_in_frame", status,
                  f"hands found in {hits}/{len(frames)} sampled phone frames (need {MIN_HAND_FRACTION:.0%})",
                  fraction=round(frac, 2))


def validate_take(session: Session, take: int, analyzers=None) -> dict:
    """`analyzers` lets tests inject {"video": fn(path), "bag": fn(path), "hands": fn(...)}."""
    az = {"video": analyze_video, "bag": analyze_bag, "hands": hands_check, **(analyzers or {})}
    devices = session.devices
    record = next((t for t in session.takes if t["take"] == take), {"take": take})
    checks, streams = [], {}

    # 1. files present
    missing = []
    for dev in devices:
        f = session.device_file(take, dev)
        if f is None:
            missing.append(dev)
            continue
        try:
            a = az["bag"](f) if dev == "realsense" else az["video"](f)
        except Exception as e:  # noqa: BLE001 - a broken file is a FAIL, not a crash
            checks.append(_check(f"{dev}_readable", "FAIL", f"{f.name}: {e}"))
            continue
        a["file"] = f.name
        a["bytes"] = f.stat().st_size
        a["sha256"] = _sha256(f)
        streams[dev] = a
    checks.append(_check("files_present", "FAIL" if missing else "PASS",
                         f"missing: {', '.join(missing)}" if missing else
                         f"all {len(devices)} devices present ({', '.join(devices)})"))

    # 2. depth present
    if "realsense" in streams:
        d = streams["realsense"]["streams"].get("depth")
        if not d or d["frames"] == 0:
            checks.append(_check("depth_present", "FAIL", "bag has no depth frames"))
        else:
            v = d.get("valid_pixel_fraction_median", 0.0)
            checks.append(_check("depth_present", "PASS" if v >= MIN_DEPTH_VALID else "FAIL",
                                 f"{d['frames']} depth frames, median valid pixels {v:.0%} "
                                 f"(need {MIN_DEPTH_VALID:.0%})"))

    # 3. frame drops
    for dev, a in streams.items():
        if dev == "realsense":
            for name, s in a["streams"].items():
                ok = s["drop_rate"] <= MAX_DROP_RATE
                checks.append(_check(f"realsense_{name}_drops", "PASS" if ok else "FAIL",
                                     f"{s['dropped_frames']} dropped of {s['expected_frames']} "
                                     f"({s['drop_rate']:.1%}, limit {MAX_DROP_RATE:.0%})"))
        else:
            nominal, eff = a["info"].get("fps"), a.get("effective_fps")
            if not nominal or not eff:
                checks.append(_check(f"{dev}_frame_rate", "SKIP", "frame rate unknown"))
            else:
                ok = eff >= MIN_FPS_RATIO * min(nominal, 30.0)
                checks.append(_check(f"{dev}_frame_rate", "PASS" if ok else "FAIL",
                                     f"effective {eff} fps vs nominal {nominal}"))

    # 4. sync event in every stream
    events = {dev: a.get("light_event") for dev, a in streams.items()}
    lost = [d for d, e in events.items() if e is None]
    if streams:
        checks.append(_check("sync_light_flick", "FAIL" if lost else "PASS",
                             f"no light flick found in: {', '.join(lost)} (first {SEARCH_S:.0f}s)"
                             if lost else "light flick found in " +
                             ", ".join(f"{d} @ {e.t:.3f}s ({e.direction}, {e.rel:.0%} change, z={e.strength})"
                                       for d, e in events.items())))
        dirs = {e.direction for e in events.values() if e}
        if len(dirs) > 1:
            checks.append(_check("sync_same_edge", "FAIL",
                                 "streams locked to different edges ("
                                 + ", ".join(f"{d}: {e.direction}" for d, e in events.items() if e)
                                 + "); one camera missed the first flick - redo the take"))

    # 4b. lighting: dark scenes still sync, but color and hand detection suffer
    for dev, a in streams.items():
        lt = a.get("lighting") or {}
        if not lt:
            continue
        status = "PASS" if lt["verdict"] == "ok" else "WARN"
        checks.append(_check(f"{dev}_lighting", status,
                             f"median brightness {lt['median']:.0f}/255 ({lt['verdict']}; "
                             f"ok range {sync.DARK:.0f}-{sync.BRIGHT:.0f})"))

    # 5. aligned durations after the flick
    tails = {d: a["duration_s"] - events[d].t for d, a in streams.items() if events.get(d)}
    if len(tails) >= 2:
        spread = max(tails.values()) - min(tails.values())
        shortest = min(tails.values())
        checks.append(_check("aligned_overlap", "PASS" if spread <= DURATION_TOL_S else "WARN",
                             f"all streams overlap for {shortest:.1f}s after the flick; "
                             f"stop times differ by {spread:.1f}s",
                             overlap_s=round(shortest, 2)))

    # 6. clap cross-check (phone vs Mac)
    pm = [d for d in ("phone", "mac") if d in streams]
    if len(pm) == 2:
        p, m = streams["phone"], streams["mac"]
        if p.get("light_event") and m.get("light_event") and p.get("clap_event") and m.get("clap_event"):
            dp = p["clap_event"].t - p["light_event"].t
            dm = m["clap_event"].t - m["light_event"].t
            err = abs(dp - dm)
            checks.append(_check("clap_crosscheck", "PASS" if err <= SYNC_TOL_S else "WARN",
                                 f"flick->clap gap phone {dp:.3f}s vs Mac {dm:.3f}s, "
                                 f"disagree by {err * 1000:.0f} ms (tolerance {SYNC_TOL_S * 1000:.0f} ms)",
                                 error_ms=round(err * 1000, 1)))
        else:
            checks.append(_check("clap_crosscheck", "WARN",
                                 "clap not found in both phone and Mac audio; alignment rests on the light flick alone"))

    # 7. hands in the first-person view
    if "phone" in streams and events.get("phone"):
        ev = events["phone"].t
        checks.append(az["hands"](session.device_file(take, "phone"), ev, streams["phone"]["duration_s"]))

    statuses = [c["status"] for c in checks]
    verdict = "FAIL" if "FAIL" in statuses else "PASS"
    warnings = statuses.count("WARN")

    # offsets: seconds to subtract from each stream's own clock so the flick is t=0
    out_streams = {}
    for dev, a in streams.items():
        e = a.pop("light_event", None)
        c = a.pop("clap_event", None)
        a["sync_offset_s"] = round(e.t, 4) if e else None
        a["light_event"] = ({"t": round(e.t, 4), "z": e.strength, "drop": e.drop,
                             "direction": e.direction, "rel": e.rel} if e else None)
        if c:
            a["clap_event"] = {"t": round(c.t, 4), "z": c.strength}
        out_streams[dev] = a
    manifest = {
        "schema": "omega-capture-take/1", "tool_version": __version__, "git_rev": _git_rev(),
        "validated_utc": utc_now(), "session": session.name, "task": session.meta.get("task", ""),
        "take": take, "recorded": record, "streams": out_streams, "checks": checks,
        "verdict": verdict, "warnings": warnings,
        "alignment": "t=0 is the first light-flick edge in every stream; subtract sync_offset_s "
                     "from that stream's time",
    }
    session.write(f"take_{take:02d}/manifest.json", manifest)
    return manifest


def validate_session(session: Session, analyzers=None) -> dict:
    takes = sorted({t["take"] for t in session.takes} |
                   {int(p.name[5:]) for p in session.path.glob("take_[0-9]*") if p.is_dir()})
    results = [validate_take(session, n, analyzers) for n in takes]
    summary = {
        "session": session.name, "task": session.meta.get("task", ""), "validated_utc": utc_now(),
        "takes": [{"take": r["take"], "verdict": r["verdict"], "warnings": r["warnings"],
                   "failed": [c["check"] for c in r["checks"] if c["status"] == "FAIL"]} for r in results],
    }
    summary["passed"] = sum(r["verdict"] == "PASS" for r in results)
    summary["failed"] = len(results) - summary["passed"]
    session.write("summary.json", summary)
    lines = [f"# Capture session `{session.name}`", "", f"Task: {summary['task'] or '-'}", "",
             f"**{summary['passed']} of {len(results)} takes PASS**", "",
             "| take | verdict | warnings | failed checks |", "|---|---|---|---|"]
    for t in summary["takes"]:
        lines.append(f"| {t['take']} | {t['verdict']} | {t['warnings']} | {', '.join(t['failed']) or '-'} |")
    (session.path / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
