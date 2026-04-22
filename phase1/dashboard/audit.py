"""Append-only JSONL audit log for save-policy decisions.

Kept small (~500 lines cap) — this is a debugging / transparency tool,
not a forensic audit trail. The dashboard UI reads the tail for the
"Recent activity" panel.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_PATH_DEFAULT = Path("~/.granaria.supermemory/dashboard_audit.jsonl").expanduser()
MAX_ENTRIES = 500

_lock = threading.Lock()
_custom_path: Path | None = None


def _path() -> Path:
    return _custom_path or AUDIT_PATH_DEFAULT


def set_path_for_tests(path: Path | str | None) -> None:
    """Redirect audit writes to an arbitrary file (for tests)."""
    global _custom_path
    _custom_path = Path(path).expanduser() if path else None


def log(event: dict[str, Any]) -> None:
    """Append one JSONL event. Failures are swallowed — audit is best-effort."""
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    try:
        p = _path()
        with _lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Never raise from audit: the calling save should continue
        pass


def recent(n: int = 50) -> list[dict[str, Any]]:
    """Return the last N events, newest-first."""
    p = _path()
    if not p.exists():
        return []
    try:
        with _lock:
            lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    out.reverse()
    return out


def trim() -> None:
    """Keep only the most recent MAX_ENTRIES lines. Idempotent."""
    p = _path()
    if not p.exists():
        return
    with _lock:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        if len(lines) <= MAX_ENTRIES:
            return
        tail = lines[-MAX_ENTRIES:]
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(tail) + "\n", encoding="utf-8")
        os.replace(tmp, p)
