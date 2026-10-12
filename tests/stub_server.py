"""Threaded local HTTP server for backend contract tests."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from time import sleep


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeError, ValueError):
            decoded = None

        state = self.server.state  # type: ignore[attr-defined]
        with state.lock:
            state.requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": raw_body,
                    "json": decoded,
                }
            )
            status, body, delay_s = state.response
        if delay_s:
            sleep(delay_s)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, format: str, *args: object) -> None:
        return


class StubServer:
    def __init__(self) -> None:
        self.lock = Lock()
        self.requests: list[dict] = []
        self.response: tuple[int, bytes, float] = (200, b"{}", 0.0)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self  # type: ignore[attr-defined]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def set_response(
        self, status: int = 200, body: bytes | str | dict | list = b"{}", delay_s: float = 0
    ) -> None:
        if isinstance(body, (dict, list)):
            encoded = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            encoded = body.encode("utf-8")
        else:
            encoded = body
        with self.lock:
            self.response = (status, encoded, delay_s)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
