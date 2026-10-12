"""Ollama's non-streaming local generate endpoint."""

from __future__ import annotations

import time
import warnings
from typing import Sequence

from tinyagent.backends.base import (
    _is_loopback_url,
    _post_json,
    _protocol_error,
    _single_line,
    _token_count,
    _validate_call,
)
from tinyagent.errors import BackendError
from tinyagent.types import Completion


class OllamaBackend:
    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 60.0,
        seed: int = 42,
        num_ctx: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.seed = seed
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"
        if not _is_loopback_url(self.host):
            warnings.warn(
                "Ollama host is non-loopback; prompts may leave this machine.",
                RuntimeWarning,
                stacklevel=2,
            )

    def complete(
        self, prompt: str, *, stop: Sequence[str], max_tokens: int
    ) -> Completion:
        stop_list = _validate_call(prompt, stop, max_tokens)
        options = {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": self.seed,
            "num_predict": max_tokens,
            "stop": stop_list,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "options": options,
        }
        started = time.perf_counter()
        response = _post_json(
            f"{self.host}/api/generate",
            payload,
            self.timeout_s,
            sensitive_text=prompt,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        server_error = response.get("error")
        if isinstance(server_error, str) and server_error.strip():
            raise BackendError(
                "E_BACKEND_HTTP", _single_line(server_error, sensitive_text=prompt)
            ) from ValueError("Ollama reported an error")

        text = response.get("response")
        if not isinstance(text, str):
            raise _protocol_error("missing or non-string response field")
        return Completion(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=_token_count(response.get("prompt_eval_count")),
            completion_tokens=_token_count(response.get("eval_count")),
            backend=self.name,
        )

    def __repr__(self) -> str:
        return (
            f"OllamaBackend(model={self.model!r}, host={self.host!r}, "
            f"timeout_s={self.timeout_s!r})"
        )
