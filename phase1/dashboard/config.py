"""Dashboard configuration — single JSON file, hot-reloadable.

Stored in `~/.granaria.supermemory/config.json`. On first access the
defaults are written; afterwards the file is the source of truth.

Hot-reload is achieved via mtime comparison on each read — negligible
overhead, no file watcher daemon needed.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

CONFIG_PATH_DEFAULT = Path("~/.granaria.supermemory/config.json").expanduser()

# Authoritative defaults. Additions here propagate to existing configs on
# next read (missing keys are filled in).
_DEFAULTS: dict[str, Any] = {
    "privacy": {
        "enabled": True,
        "strip_private_tags": True,
        "strict_unclosed_tags": True,
        "patterns": {
            "aws_access_key": True,
            "stripe_live_key": True,
            "stripe_test_key": True,
            "github_pat": True,
            "github_oauth": True,
            "openai_key": True,
            "anthropic_key": True,
            "jwt": True,
            "private_key_pem": True,
        },
    },
    "capture": {
        "max_content_chars": 200000,
        "blocked_projects": [],
        "auto_extract_graph": True,
        "default_language": "auto",  # "de" | "en" | "auto"
    },
}


def _deep_merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return defaults with override recursively merged in."""
    out = copy.deepcopy(defaults)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Thread-safe config holder with mtime-based hot-reload."""

    def __init__(self, path: Path | str = CONFIG_PATH_DEFAULT):
        self._path = Path(path).expanduser()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._mtime: float = 0.0
        self._ensure_loaded()

    # ── Internal ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._data = copy.deepcopy(_DEFAULTS)
                self._save_unlocked()
            else:
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except Exception:
                    # Corrupt config — fall back to defaults without clobbering
                    # the file; caller sees defaults, can fix and save again.
                    self._data = copy.deepcopy(_DEFAULTS)
                    return
                self._data = _deep_merge(_DEFAULTS, raw)
            try:
                self._mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                self._mtime = 0.0

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    def _reload_if_stale(self) -> None:
        try:
            m = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if m != self._mtime:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = _deep_merge(_DEFAULTS, raw)
                self._mtime = m
            except Exception:
                # Keep current in-memory data if reload fails
                pass

    # ── Public API ───────────────────────────────────────────────

    def get(self) -> dict[str, Any]:
        """Return a deep copy of the current config (hot-reloaded)."""
        with self._lock:
            self._reload_if_stale()
            return copy.deepcopy(self._data)

    def save(self, new_data: dict[str, Any]) -> dict[str, Any]:
        """Merge `new_data` into defaults, persist, return the merged result."""
        with self._lock:
            self._data = _deep_merge(_DEFAULTS, new_data)
            self._save_unlocked()
            try:
                self._mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                self._mtime = 0.0
            return copy.deepcopy(self._data)

    @property
    def path(self) -> Path:
        return self._path


# ── Module-level singleton (lazy) ─────────────────────────────────

_singleton: Config | None = None


def get_config() -> Config:
    global _singleton
    if _singleton is None:
        _singleton = Config()
    return _singleton


def reset_for_tests(path: Path | str | None = None) -> Config:
    """Replace the singleton — for unit tests with a tmp config file."""
    global _singleton
    _singleton = Config(path) if path else Config()
    return _singleton
