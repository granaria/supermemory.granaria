"""Minimal HTTP server for the supermemory.granaria dashboard.

Pure stdlib — no flask/fastapi — so there's zero new dependencies and
nothing weird happens at startup. Serves a single HTML page + JSON API.
Binds strictly to 127.0.0.1; this is not meant to be exposed.

Run:
    python -m phase1.dashboard

Endpoints:
    GET  /              → dashboard HTML
    GET  /api/config    → current config (JSON)
    POST /api/config    → merge+save new config (JSON body) → updated config
    GET  /api/stats     → memory/chunk/project counts + graph stats
    GET  /api/audit     → recent save-policy events (JSONL tail)
    GET  /api/projects  → distinct project names (for blocklist UI)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import get_config
from . import audit

log = logging.getLogger("supermemory.dashboard")

HERE = Path(__file__).parent
INDEX_HTML = HERE / "index.html"

HOST = os.environ.get("GRANARIA_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("GRANARIA_DASHBOARD_PORT", "7333"))


# ── Stats + projects helpers ──────────────────────────────────────

def _load_stats() -> dict[str, Any]:
    """Best-effort read of MemoryStore stats. Returns {} on failure."""
    try:
        # Import lazily: the dashboard works standalone even if the MCP
        # server/store import chain is broken.
        from local_supermemory.store import MemoryStore
        s = MemoryStore()
        return s.stats()
    except Exception as e:
        log.warning("stats unavailable: %s", e)
        return {"error": str(e)}


def _list_projects() -> list[str]:
    try:
        from local_supermemory.store import MemoryStore
        s = MemoryStore()
        return sorted([p["name"] for p in s.list_projects()])
    except Exception:
        return []


# ── HTTP handler ──────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    server_version = "granaria-dashboard/1.0"

    def log_message(self, fmt, *args):  # noqa: N802 — stdlib override
        # Quieter console output; forward to logger if needed
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # ── Routing ───────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send_html(INDEX_HTML.read_bytes())
            except FileNotFoundError:
                self._send_html(b"<h1>index.html missing</h1>", 500)
            return
        if path == "/api/config":
            self._send_json(get_config().get())
            return
        if path == "/api/stats":
            self._send_json(_load_stats())
            return
        if path == "/api/audit":
            n = 50
            # ?n=<int>
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for part in q.split("&"):
                    if part.startswith("n="):
                        try:
                            n = max(1, min(500, int(part[2:])))
                        except ValueError:
                            pass
            self._send_json({"events": audit.recent(n)})
            return
        if path == "/api/projects":
            self._send_json({"projects": _list_projects()})
            return
        self._send_json({"error": "not_found", "path": path}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/config":
            try:
                payload = self._read_json_body()
            except Exception as e:
                self._send_json({"error": f"invalid JSON: {e}"}, 400)
                return
            updated = get_config().save(payload)
            audit.log({"event": "config_updated"})
            self._send_json(updated)
            return
        self._send_json({"error": "not_found", "path": path}, 404)


# ── Entrypoint ────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=os.environ.get("GRANARIA_DASHBOARD_LOG", "INFO"),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    # Touch config to write defaults if missing
    cfg = get_config()
    srv = ThreadingHTTPServer((HOST, PORT), _Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"supermemory.granaria dashboard")
    print(f"  URL:    {url}")
    print(f"  Config: {cfg.path}")
    print(f"  Audit:  {audit.AUDIT_PATH_DEFAULT}")
    print(f"  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping dashboard…")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
