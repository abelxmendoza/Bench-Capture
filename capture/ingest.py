"""Collect phone / MacBook clips and file each one under the take it belongs to.

Clips arrive in <session>/inbox/ - by Taildrop (`tailscale file get`, run here) or
copied in by hand. Each clip is matched to a take by its recording time overlapping
the take's window, and to a device by its metadata (an iPhone writes its model).
"""
from __future__ import annotations

import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

from . import media
from .session import VIDEO_EXT, Session, parse_utc

MARGIN = timedelta(seconds=90)   # phone/Mac are started by hand around the RealSense window


def pull_taildrop(inbox: Path) -> list[str]:
    """Receive any files waiting in Taildrop. Returns names received (best effort)."""
    if shutil.which("tailscale") is None:
        return []
    before = {p.name for p in inbox.iterdir()}
    r = subprocess.run(["tailscale", "file", "get", "--conflict=rename", "--wait=false",
                        str(inbox)], capture_output=True, text=True, check=False)
    if r.returncode != 0 and "no files" not in (r.stderr + r.stdout).lower():
        print(f"  taildrop: {(r.stderr or r.stdout).strip()[:200]}")
    return sorted({p.name for p in inbox.iterdir()} - before)


def classify(path: Path, info: dict) -> str | None:
    name = path.name.lower()
    if "phone" in name or "iphone" in name:
        return "phone"
    if "mac" in name:
        return "mac"
    model = f"{info.get('model') or ''} {info.get('make') or ''}".lower()
    if "iphone" in model or "android" in model or "pixel" in model or "samsung" in model:
        return "phone"
    # QuickTime on a Mac writes no device model; a clip with none is the MacBook.
    if not info.get("model"):
        return "mac"
    return None


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())


def match_take(info: dict, takes: list[dict]) -> int | None:
    ct = info.get("creation_time")
    if not ct or not takes:
        return None
    try:
        c0 = parse_utc(ct)
    except ValueError:
        return None
    dur = timedelta(seconds=info.get("duration_s") or 0)
    # Some cameras stamp the start, some the end; test both readings.
    windows = [(c0, c0 + dur), (c0 - dur, c0)]
    best, best_s = None, 0.0
    for t in takes:
        if "start_utc" not in t or "end_utc" not in t:
            continue
        t0, t1 = parse_utc(t["start_utc"]) - MARGIN, parse_utc(t["end_utc"]) + MARGIN
        s = max(_overlap(w0, w1, t0, t1) for w0, w1 in windows)
        if s > best_s:
            best, best_s = t["take"], s
    return best


def ingest(session: Session, taildrop: bool = True, assign: dict | None = None) -> list[dict]:
    """File inbox clips into take folders. `assign` = {"phone": (take, path), ...} overrides."""
    inbox = session.path / "inbox"
    if taildrop:
        got = pull_taildrop(inbox)
        if got:
            print(f"  taildrop: received {', '.join(got)}")
    results = []
    jobs = []
    for device, (take, path) in (assign or {}).items():
        jobs.append((Path(path), device, take))
    for p in sorted(inbox.iterdir()):
        if p.suffix.lower() in VIDEO_EXT:
            jobs.append((p, None, None))
    takes = session.takes
    for path, device, take in jobs:
        rec = {"file": path.name}
        try:
            info = media.probe(path)
        except RuntimeError as e:
            rec.update(status="unreadable", detail=str(e)[:200])
            results.append(rec)
            continue
        device = device or classify(path, info)
        take = take or match_take(info, takes)
        rec.update(device=device, take=take, creation_time=info.get("creation_time"),
                   model=info.get("model"))
        if device is None or take is None:
            rec["status"] = "unmatched (left in inbox)"
            results.append(rec)
            continue
        dest = session.take_dir(take) / f"{device}{path.suffix.lower()}"
        if dest.exists() and dest.resolve() != path.resolve():
            rec["status"] = f"skipped: take {take} already has {dest.name}"
            results.append(rec)
            continue
        shutil.move(str(path), dest) if path.parent == inbox else shutil.copy2(path, dest)
        rec.update(status="filed", dest=str(dest.relative_to(session.path)))
        results.append(rec)
    return results
