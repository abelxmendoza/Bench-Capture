"""Log each validated session into agent-hivemind so the other agents see it.

Appends one row per session to <hivemind>/state/capture_sessions.csv (replacing an
earlier row for the same session). With push=True it commits and pushes that one file.
"""
from __future__ import annotations

import csv
import os
import socket
import subprocess
from pathlib import Path

from .session import utc_now

FIELDS = ["logged_utc", "session", "task", "takes", "passed", "failed", "host", "path"]


def default_dir() -> Path | None:
    env = os.environ.get("HIVEMIND_DIR")
    if env:
        return Path(env).expanduser()
    for p in (Path.home() / "agent-hivemind", Path.home() / "Desktop" / "agent-hivemind"):
        if (p / ".git").exists():
            return p
    return None


def log_session(summary: dict, session_path: Path, hivemind: Path | None = None,
                push: bool = False) -> str:
    hivemind = hivemind or default_dir()
    if hivemind is None or not hivemind.is_dir():
        return "hivemind: no clone found (set HIVEMIND_DIR); skipped"
    out = hivemind / "state" / "capture_sessions.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    if out.is_file():
        with out.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("session") != summary["session"]]
    rows.append({"logged_utc": utc_now(), "session": summary["session"], "task": summary.get("task", ""),
                 "takes": len(summary["takes"]), "passed": summary["passed"],
                 "failed": summary["failed"], "host": socket.gethostname(), "path": str(session_path)})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    msg = f"hivemind: logged to {out}"
    if push:
        git = ["git", "-C", str(hivemind)]
        steps = [git + ["add", str(out.relative_to(hivemind))],
                 git + ["commit", "-m", f"capture: {summary['session']} "
                        f"({summary['passed']}/{len(summary['takes'])} takes PASS)", "--",
                        str(out.relative_to(hivemind))],
                 git + ["pull", "--rebase", "--autostash", "-q"],
                 git + ["push", "-q"]]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return f"{msg}; push stopped at `{' '.join(cmd[3:5])}`: {(r.stderr or r.stdout).strip()[:200]}"
        msg += " and pushed"
    return msg
