"""Session layout on disk.

    $CAPTURE_ROOT/<session>/
        session.json            task, devices, created
        takes.json              one record per take (start/end UTC, devices)
        inbox/                  phone / MacBook clips land here (Taildrop or by hand)
        take_01/
            realsense.bag
            phone.mov | phone.mp4
            mac.mov   | mac.mp4
            manifest.json       written by `validate`
        summary.json / summary.md
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("CAPTURE_ROOT", Path.home() / "captures"))
DEVICES = ("realsense", "phone", "mac")
VIDEO_EXT = (".mov", ".mp4", ".m4v")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@dataclass
class Session:
    path: Path

    @classmethod
    def open(cls, name_or_path: str, root: Path = DEFAULT_ROOT, task: str | None = None,
             devices=DEVICES) -> "Session":
        p = Path(name_or_path)
        if not (p.is_absolute() or p.exists()):
            if not re.fullmatch(r"[A-Za-z0-9._-]+", name_or_path):
                raise ValueError(f"session name must be letters, digits, . _ - : {name_or_path!r}")
            p = Path(root) / name_or_path
        p.mkdir(parents=True, exist_ok=True)
        (p / "inbox").mkdir(exist_ok=True)
        s = cls(p)
        meta = s.meta
        if not meta:
            meta = {"created_utc": utc_now(), "task": task or "", "devices": list(devices)}
        elif task:
            meta["task"] = task
        _write_json(p / "session.json", meta)
        return s

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def meta(self) -> dict:
        return _read_json(self.path / "session.json", {})

    @property
    def devices(self) -> list[str]:
        return list(self.meta.get("devices", DEVICES))

    @property
    def takes(self) -> list[dict]:
        return _read_json(self.path / "takes.json", [])

    def take_dir(self, n: int) -> Path:
        d = self.path / f"take_{n:02d}"
        d.mkdir(exist_ok=True)
        return d

    def next_take(self) -> int:
        return max([t["take"] for t in self.takes], default=0) + 1

    def record_take(self, record: dict) -> None:
        takes = [t for t in self.takes if t["take"] != record["take"]]
        takes.append(record)
        _write_json(self.path / "takes.json", sorted(takes, key=lambda t: t["take"]))

    def device_file(self, n: int, device: str) -> Path | None:
        d = self.path / f"take_{n:02d}"
        if device == "realsense":
            f = d / "realsense.bag"
            return f if f.is_file() else None
        for ext in VIDEO_EXT:
            f = d / f"{device}{ext}"
            if f.is_file():
                return f
        return None

    def write(self, rel: str, data) -> Path:
        out = self.path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, data)
        return out
