"""llama.cpp's non-streaming completion endpoint."""

from __future__ import annotations

import time
import warnings
from typing import Sequence

from tinyagent.backends.base import (
    _is_loopback_url,
    _post_json,
    _protocol_error,
    _token_count,
    _validate_call,
)
from tinyagent.types import Completion


class LlamaCppBackend:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        timeout_s: float = 60.0,
        seed: int = 42,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.seed = seed
        self.name = "llamacpp"
        if not _is_loopback_url(self.base_url):
            warnings.warn(
                "llama.cpp host is non-loopback; prompts may leave this machine.",
                RuntimeWarning,
                stacklevel=2,
            )

    def complete(
        self, prompt: str, *, stop: Sequence[str], max_tokens: int
    ) -> Completion:
        stop_list = _validate_call(prompt, stop, max_tokens)
        payload = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": self.seed,
            "n_predict": max_tokens,
            "stop": stop_list,
            "cache_prompt": True,
        }
        started = time.perf_counter()
        response = _post_json(
            f"{self.base_url}/completion",
            payload,
            self.timeout_s,
            sensitive_text=prompt,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        text = response.get("content")
        if not isinstance(text, str):
            raise _protocol_error("missing or non-string content field")
        return Completion(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=_token_count(response.get("tokens_evaluated")),
            completion_tokens=_token_count(response.get("tokens_predicted")),
            backend=self.name,
        )

    def __repr__(self) -> str:
        return f"LlamaCppBackend(base_url={self.base_url!r}, timeout_s={self.timeout_s!r})"
