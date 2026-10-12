"""Allowlisted executor: ParsedAction -> ToolResult (PRD-03 Section 5).

Checks run in a fixed order and the first failure returns ``ok=False`` without
calling the tool. A tool is resolved only by exact name through
``ToolRegistry.get``; there is no fuzzy matching or fallback. Each call runs the
tool at most once and never retries, because tools may have side effects.

Timeout limitation (R3-63): the tool runs in a daemon worker thread. Python
cannot kill a thread, so after ``E_TOOL_TIMEOUT`` the tool may keep running in
the background until it returns. Tools that touch external systems must
enforce their own timeouts.
"""

from __future__ import annotations

import math
import re
import threading
import time
from typing import Any

from tinyagent.coercion import coerce
from tinyagent.errors import CoercionError
from tinyagent.registry import ToolRegistry
from tinyagent.types import ParsedAction, ToolResult, ToolSpec

TRUNCATION_MARKER = "...[truncated]"

# Absolute POSIX paths with 2+ components, and Windows drive paths (R3-62).
_PATH = re.compile(r"(?<![\w.])(?:/[\w.-]+){2,}/?|\b[A-Za-z]:\\[^\s'\"]*")


def _single_line(value: object, limit: int) -> str:
    text = _PATH.sub("[path]", " ".join(str(value).split()))
    return text[:limit]


def _bind(action: ParsedAction, spec: ToolSpec) -> tuple[dict[str, str] | None, str, str]:
    """Steps 2 to 4: map raw args onto parameter names.

    Returns (bound, "", "") on success or (None, code, detail) on failure.
    """
    params = spec.params
    names = [param.name for param in params]
    keywords = {key: value for key, value in action.args.items() if not _is_positional(key)}
    bound: dict[str, str] = {}
    for key, value in action.args.items():
        if not _is_positional(key):
            continue
        index = int(key[1:])
        if index >= len(params):
            return None, "E_UNEXPECTED_ARG", key  # R3-51 too many positionals
        name = names[index]
        if name in keywords:
            return None, "E_UNEXPECTED_ARG", name  # R3-51 collides with keyword
        bound[name] = value
    for key, value in keywords.items():
        if key not in names:
            return None, "E_UNEXPECTED_ARG", _single_line(key, 40)  # R3-52
        bound[key] = value
    for param in params:
        if param.required and param.name not in bound:
            return None, "E_MISSING_ARG", param.name  # R3-53
    return bound, "", ""


def _is_positional(key: str) -> bool:
    return key.startswith("#") and key[1:].isascii() and key[1:].isdigit()


def _run_with_timeout(func, kwargs: dict[str, Any], timeout_s: float) -> tuple[str, Any]:
    """Step 6 to 8 inside one worker: ("ok", text) | ("error", exc) | ("timeout", None)."""
    outcome: list[tuple[str, Any]] = []

    def worker() -> None:
        try:
            result = func(**kwargs)
            outcome.append(("ok", "" if result is None else str(result)))  # R3-61
        except BaseException as exc:  # noqa: BLE001 - contained and reported, never re-raised
            outcome.append(("error", exc))

    thread = threading.Thread(target=worker, name="tinyagent-tool", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive() or not outcome:
        return "timeout", None
    return outcome[0]


def _validate_limits(timeout_s: float, max_output_chars: int) -> None:
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive finite number")
    if type(max_output_chars) is not int or max_output_chars < 1:
        raise ValueError("max_output_chars must be a positive int")


def execute(
    action: ParsedAction,
    registry: ToolRegistry,
    *,
    timeout_s: float = 10.0,
    max_output_chars: int = 4000,
) -> ToolResult:
    """Validate, coerce and run one tool call. Never raises for model output."""
    _validate_limits(timeout_s, max_output_chars)
    started = time.perf_counter()

    def failure(code: str, detail: str, args: dict[str, Any] | None = None) -> ToolResult:
        elapsed = (time.perf_counter() - started) * 1000
        return ToolResult(False, action.tool, args or {}, None, code, detail, elapsed)

    spec = registry.get(action.tool)  # R3-50, S3-2: exact match only
    if spec is None:
        return failure("E_UNKNOWN_TOOL", _single_line(action.tool, 40))

    bound, code, detail = _bind(action, spec)
    if bound is None:
        return failure(code, detail)

    kwargs: dict[str, Any] = {}
    for param in spec.params:  # R3-54
        if param.name in bound:
            try:
                kwargs[param.name] = coerce(bound[param.name], param)
            except CoercionError:
                return failure("E_TYPE_COERCION", param.name)
        else:
            kwargs[param.name] = param.default

    status, value = _run_with_timeout(spec.func, kwargs, timeout_s)  # R3-55, R3-60
    if status == "timeout":
        return failure("E_TOOL_TIMEOUT", f"exceeded {timeout_s:g}s", kwargs)
    if status == "error":  # R3-56
        return failure(
            "E_TOOL_EXCEPTION", _single_line(f"{type(value).__name__}: {value}", 200), kwargs
        )

    output = value
    if len(output) > max_output_chars:  # R3-57
        output = output[:max_output_chars] + TRUNCATION_MARKER
    elapsed = (time.perf_counter() - started) * 1000
    return ToolResult(True, action.tool, kwargs, output, None, None, elapsed)
