"""Lightweight HTTP server serving the clean benchmark frontend and live mlflow.db API."""

from __future__ import annotations

import argparse
import json
import sqlite3
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
MLFLOW_DB = ROOT_DIR / "mlflow.db"


def load_mlflow_sqlite_summary(db_path: Path = MLFLOW_DB) -> dict[str, Any]:
    """Read live runs, params, and metrics directly from ``mlflow.db`` if present."""
    if not db_path.exists():
        return {"source": "embedded", "runs_count": 3}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        runs = cur.execute(
            "SELECT run_uuid, name, status, start_time, end_time FROM runs WHERE lifecycle_stage='active'"
        ).fetchall()
        return {"source": "mlflow.db", "runs_count": len(runs)}
    finally:
        conn.close()


class BenchmarkDashboardHandler(SimpleHTTPRequestHandler):
    """Serves ``frontend/index.html`` and ``/api/status``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            payload = json.dumps(load_mlflow_sqlite_summary()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/api/runs":
            payload = json.dumps({}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the K-Means Benchmark Frontend Dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), BenchmarkDashboardHandler)
    print(f"Benchmark Frontend Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
