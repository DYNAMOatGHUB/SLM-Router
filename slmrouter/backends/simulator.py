"""Deterministic backends for offline evaluation and exact replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Callable, Sequence

from slmrouter.types import Backend, Completion


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class SimulatedBackend:
    def __init__(
        self,
        script: Sequence[str] | Callable[[str, int], str],
        *,
        name: str = "sim",
    ) -> None:
        self.name = name
        self._script = script
        self.prompts: list[str] = []
        self._lock = Lock()

    def complete(
        self, prompt: str, *, stop: Sequence[str], max_tokens: int
    ) -> Completion:
        from slmrouter.prompt import estimate_tokens

        with self._lock:
            index = len(self.prompts)
            self.prompts.append(prompt)
            if callable(self._script):
                text = self._script(prompt, index)
            else:
                if index >= len(self._script):
                    raise AssertionError("simulator script exhausted")
                text = self._script[index]
        return Completion(
            text, 0.0, estimate_tokens(prompt), estimate_tokens(text), self.name
        )


class RecordingBackend:
    def __init__(self, inner: Backend, path: str | Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self._lock = Lock()

    def complete(
        self, prompt: str, *, stop: Sequence[str], max_tokens: int
    ) -> Completion:
        completion = self.inner.complete(prompt, stop=stop, max_tokens=max_tokens)
        record = {
            "prompt_sha256": _hash(prompt),
            "prompt": prompt,
            "text": completion.text,
            "backend": completion.backend,
            "latency_ms": completion.latency_ms,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return completion


class ReplayBackend:
    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self._records: dict[str, list[dict]] = {}
        self._lock = Lock()
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                digest = record["prompt_sha256"]
                if _hash(record["prompt"]) != digest:
                    raise ValueError(f"fixture prompt hash mismatch: {digest}")
                self._records.setdefault(digest, []).append(record)

    def complete(
        self, prompt: str, *, stop: Sequence[str], max_tokens: int
    ) -> Completion:
        digest = _hash(prompt)
        with self._lock:
            queue = self._records.get(digest)
            if not queue:
                raise KeyError(digest)
            record = queue.pop(0)
        return Completion(
            record["text"],
            record["latency_ms"],
            record["prompt_tokens"],
            record["completion_tokens"],
            record["backend"],
        )
