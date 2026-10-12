"""Shared standard-library HTTP transport for local inference backends."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from tinyagent.errors import BackendError


def _single_line(value: object, *, sensitive_text: str | None = None) -> str:
    text = str(value)
    if sensitive_text:
        text = text.replace(sensitive_text, "[redacted]")
    return " ".join(text.split())[:200]


def _is_loopback_url(url: str) -> bool:
    authority = url.split("://", 1)[-1].split("/", 1)[0]
    authority = authority.split("?", 1)[0].split("#", 1)[0].rsplit("@", 1)[-1]
    if authority.startswith("["):
        host = authority[1:].split("]", 1)[0]
    elif authority.count(":") == 1:
        host = authority.rsplit(":", 1)[0]
    else:
        host = authority

    if host.lower() == "localhost":
        return True
    try:
        return socket.inet_pton(socket.AF_INET, host)[0] == 127
    except OSError:
        pass
    try:
        return socket.inet_pton(socket.AF_INET6, host) == (b"\x00" * 15 + b"\x01")
    except OSError:
        return False


def _http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read(200)
    except Exception:
        return ""
    return body.decode("utf-8", errors="replace")


def _post_json(
    url: str,
    payload: dict,
    timeout_s: float,
    *,
    sensitive_text: str | None = None,
) -> dict:
    """POST one UTF-8 JSON request and return a decoded JSON object."""
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw_response = response.read()
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}: {_http_error_body(exc)}"
        raise BackendError(
            "E_BACKEND_HTTP", _single_line(detail, sensitive_text=sensitive_text)
        ) from exc
    except Exception as exc:
        # URLError wraps the underlying socket failure in .reason.
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, (socket.timeout, TimeoutError)):
            code, detail = "E_BACKEND_TIMEOUT", f"no response within {timeout_s}s"
        else:
            code, detail = "E_BACKEND_UNREACHABLE", f"cannot reach {url}: {reason}"
        raise BackendError(
            code, _single_line(detail, sensitive_text=sensitive_text)
        ) from exc

    try:
        result = json.loads(raw_response.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise _protocol_error("invalid JSON") from exc
    if not isinstance(result, dict):
        raise _protocol_error("expected an object")
    return result


def _validate_call(prompt: str, stop: object, max_tokens: int) -> list[str]:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("max_tokens must be an int in 1..512")
    try:
        return [str(value) for value in stop]  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("stop must be a sequence of strings") from exc


def _token_count(value: object) -> int | None:
    return value if type(value) is int else None


def _protocol_error(message: str) -> BackendError:
    error = BackendError(
        "E_BACKEND_PROTOCOL", _single_line(f"unexpected response shape: {message}")
    )
    error.__cause__ = ValueError(message)
    return error
