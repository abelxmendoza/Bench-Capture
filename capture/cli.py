"""python -m capture <command>

  devices                          list RealSense cameras (and their USB type)
  light-check [--seconds N]        live brightness readout: set up lights, practice the flick
  take SESSION [--task T]          record one take (RealSense here; phone + Mac by hand)
  ingest SESSION                   pull Taildrop clips, file them under their takes
  validate SESSION [--take N]      checks -> take_NN/manifest.json, summary.md
  log SESSION [--push]             record the session in agent-hivemind
  run SESSION [--push]             ingest + validate + log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import hivemind, ingest as ingest_mod, validate as validate_mod
from .session import DEFAULT_ROOT, DEVICES, Session, utc_now


def _session(a) -> Session:
    devices = tuple(a.devices.split(",")) if getattr(a, "devices", None) else DEVICES
    bad = set(devices) - set(DEVICES)
    if bad:
        raise SystemExit(f"unknown device(s): {', '.join(sorted(bad))} (choose from {', '.join(DEVICES)})")
    return Session.open(a.session, root=Path(a.root), task=getattr(a, "task", None), devices=devices)


def _wait(prompt: str, duration: float | None) -> None:
    if duration is not None:
        print(f"{prompt} [auto: {duration:.0f}s]", flush=True)
        time.sleep(duration)
    else:
        input(f"{prompt} ")


def cmd_devices(a) -> int:
    from . import realsense  # noqa: PLC0415
    found = realsense.devices()
    if not found:
        print("no RealSense camera found - plug it into a USB 3 port")
        return 1
    for d in found:
        print(json.dumps(d))
    return 0


def cmd_light_check(a) -> int:
    from . import lightcheck  # noqa: PLC0415
    print(f"Live RealSense brightness for {a.seconds:.0f}s - aim the camera, adjust the lights, "
          "practice the flick. Nothing is recorded.\n")
    s = lightcheck.run(a.seconds, a.width, a.height, a.fps, a.serial, snapshot=a.snapshot)
    return 0 if s["first_flick"] else 2


def cmd_take(a) -> int:
    s = _session(a)
    n = a.take or s.next_take()
    d = s.take_dir(n)
    manual = [x for x in ("phone", "mac") if x in s.devices]
    print(f"\n=== {s.name} / take {n:02d} ===")
    if manual:
        names = {"phone": "phone (chest, 1080p30, wide)", "mac": "MacBook QuickTime (File > New Movie Recording)"}
        print("Start recording on: " + "; ".join(names[x] for x in manual))
        if not a.yes:
            input("Press Enter once they are all recording... ")
    rec = None
    start = utc_now()
    if "realsense" in s.devices:
        from . import realsense  # noqa: PLC0415
        bag = d / "realsense.bag"
        if bag.exists() and not a.overwrite:
            raise SystemExit(f"{bag} exists; use --take with a new number or --overwrite")
        rec = realsense.Recorder(bag, a.width, a.height, a.fps, a.serial)
        rec.start()
        start = utc_now()
        print(f"RealSense recording -> {bag}  ({rec.device.get('name')}, USB {rec.device.get('usb')})")
    print("\nNOW: light flick (lamp off ~1 s then on, or flashlight on ~1 s then off), "
          "clap once, then do the task.")
    try:
        _wait("Press Enter to stop...", a.duration)
    except KeyboardInterrupt:
        print()
    end = utc_now()
    if rec:
        rec.stop()
        if rec.error:
            print(f"WARNING: RealSense reported an error while recording: {rec.error}")
    record = {"take": n, "start_utc": start, "end_utc": end, "devices": s.devices,
              "realsense_frames": rec.frames if rec else None,
              "realsense_device": rec.device if rec else None}
    s.record_take(record)
    print(f"\nTake {n:02d} saved ({end[11:19]} UTC).")
    if manual:
        print("Stop the phone and MacBook recordings, then send both clips with "
              "Share > Taildrop > scythe-1 (or copy them into " f"{s.path / 'inbox'}).")
    print(f"When all clips are sent: python -m capture run {s.name}")
    return 0


def cmd_ingest(a) -> int:
    s = _session(a)
    assign = {}
    for dev in ("phone", "mac"):
        p = getattr(a, dev, None)
        if p:
            if not a.take:
                raise SystemExit(f"--{dev} needs --take")
            assign[dev] = (a.take, p)
    res = ingest_mod.ingest(s, taildrop=not a.no_taildrop, assign=assign)
    if not res:
        print("  inbox empty - nothing to file")
    for r in res:
        print(f"  {r['file']}: {r['status']}" + (f" -> {r['dest']}" if r.get("dest") else "")
              + (f" [{r.get('device')}, take {r.get('take')}]" if r["status"].startswith("unmatched") else ""))
    return 0


def _print_manifest(m: dict) -> None:
    print(f"\ntake {m['take']:02d}: {m['verdict']}" + (f" ({m['warnings']} warning(s))" if m["warnings"] else ""))
    for c in m["checks"]:
        print(f"  [{c['status']:4}] {c['check']}: {c['detail']}")


def cmd_validate(a) -> int:
    s = _session(a)
    if a.take:
        m = validate_mod.validate_take(s, a.take)
        _print_manifest(m)
        return 0 if m["verdict"] == "PASS" else 2
    summary = validate_mod.validate_session(s)
    for t in summary["takes"]:
        _print_manifest(json.loads((s.path / f"take_{t['take']:02d}" / "manifest.json").read_text()))
    print(f"\n{summary['passed']}/{len(summary['takes'])} takes PASS - {s.path / 'summary.md'}")
    return 0 if summary["failed"] == 0 else 2


def cmd_log(a) -> int:
    s = _session(a)
    f = s.path / "summary.json"
    if not f.is_file():
        raise SystemExit("no summary.json yet - run validate first")
    print(hivemind.log_session(json.loads(f.read_text()), s.path,
                               Path(a.hivemind) if a.hivemind else None, push=a.push))
    return 0


def cmd_run(a) -> int:
    print("[1/3] ingest")
    a.phone = a.mac = None
    a.take = None
    cmd_ingest(a)
    print("[2/3] validate")
    rc = cmd_validate(a)
    print("[3/3] log")
    cmd_log(a)
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m capture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="sessions root (default $CAPTURE_ROOT or ~/captures)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices").set_defaults(func=cmd_devices)

    lc = sub.add_parser("light-check", help="live brightness readout for setting up lights")
    lc.add_argument("--seconds", type=float, default=30.0)
    lc.add_argument("--width", type=int, default=848)
    lc.add_argument("--height", type=int, default=480)
    lc.add_argument("--fps", type=int, default=30)
    lc.add_argument("--serial")
    lc.add_argument("--snapshot", help="save a color|depth PNG of what the camera sees, e.g. view.png")
    lc.set_defaults(func=cmd_light_check)

    def with_session(name, func):
        sp = sub.add_parser(name)
        sp.add_argument("session")
        sp.add_argument("--devices", help=f"comma list, default {','.join(DEVICES)}")
        sp.set_defaults(func=func)
        return sp

    t = with_session("take", cmd_take)
    t.add_argument("--task", help="what is being done, e.g. 'pick and place 3 bolts'")
    t.add_argument("--take", type=int, help="take number (default: next)")
    t.add_argument("--duration", type=float, help="stop after N seconds instead of on Enter")
    t.add_argument("--yes", action="store_true", help="don't wait for Enter before recording")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--width", type=int, default=848)
    t.add_argument("--height", type=int, default=480)
    t.add_argument("--fps", type=int, default=30)
    t.add_argument("--serial", help="RealSense serial if more than one is attached")

    i = with_session("ingest", cmd_ingest)
    i.add_argument("--no-taildrop", action="store_true")
    i.add_argument("--take", type=int, help="with --phone/--mac: file that clip under this take")
    i.add_argument("--phone", help="path to a phone clip (skips auto-matching)")
    i.add_argument("--mac", help="path to a MacBook clip (skips auto-matching)")

    v = with_session("validate", cmd_validate)
    v.add_argument("--take", type=int)

    for name, func in (("log", cmd_log), ("run", cmd_run)):
        sp = with_session(name, func)
        sp.add_argument("--hivemind", help="agent-hivemind clone (default $HIVEMIND_DIR or ~/agent-hivemind)")
        sp.add_argument("--push", action="store_true", help="commit + push the log row")
        if name == "run":
            sp.add_argument("--no-taildrop", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
