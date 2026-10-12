"""llama.cpp server backend (non-streaming).

By default prompts go to the OpenAI-compatible ``/v1/chat/completions``
endpoint, where the server wraps them in the model's chat template, matching
Ollama's default (``raw: false``). Instruction-tuned models need this: with
mistral-7b-instruct, raw ``/completion`` prompts chose the right tool for 27/50
benchmark queries versus 50/50 with the template. Pass ``chat_template=False``
to send the raw prompt to ``/completion`` for base (non-instruct) models.
"""

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
        chat_template: bool = True,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(chat_template, bool):
            raise ValueError("chat_template must be a bool")
        self.chat_template = chat_template
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
        sampling = {
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": self.seed,
            "stop": stop_list,
            "cache_prompt": True,
        }
        if self.chat_template:
            path = "/v1/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                **sampling,
            }
        else:
            path = "/completion"
            payload = {"prompt": prompt, "n_predict": max_tokens, **sampling}
        started = time.perf_counter()
        response = _post_json(
            f"{self.base_url}{path}",
            payload,
            self.timeout_s,
            sensitive_text=prompt,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if self.chat_template:
            text, prompt_tokens, completion_tokens = _read_chat(response)
        else:
            text = response.get("content")
            if not isinstance(text, str):
                raise _protocol_error("missing or non-string content field")
            prompt_tokens = _token_count(response.get("tokens_evaluated"))
            completion_tokens = _token_count(response.get("tokens_predicted"))
        return Completion(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            backend=self.name,
        )

    def __repr__(self) -> str:
        return (
            f"LlamaCppBackend(base_url={self.base_url!r}, timeout_s={self.timeout_s!r}, "
            f"chat_template={self.chat_template!r})"
        )


def _read_chat(response: dict) -> tuple[str, int | None, int | None]:
    """Extract text and token counts from an OpenAI-style chat completion."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _protocol_error("missing choices")
    message = choices[0].get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise _protocol_error("missing or non-string message content")
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return (
        text,
        _token_count(usage.get("prompt_tokens")),
        _token_count(usage.get("completion_tokens")),
    )
