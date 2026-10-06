"""Bounded, single-decision orchestration for TinyAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyagent.registry import ToolRegistry
    from tinyagent.types import AgentResult, Backend, ToolResult


class TinyAgent:
    def __init__(
        self,
        backend: Backend,
        registry: ToolRegistry | None = None,
        *,
        max_repairs: int = 1,
        require_tool: bool = False,
        max_tokens: int = 96,
        tool_timeout_s: float = 10.0,
    ) -> None:
        from tinyagent.registry import default_registry

        if max_repairs not in (0, 1, 2):
            raise ValueError("max_repairs must be 0, 1, or 2")
        self.backend = backend
        self.registry = default_registry if registry is None else registry
        if not self.registry.all():
            raise ValueError("no tools registered")
        self.max_repairs = max_repairs
        self.require_tool = require_tool
        self.max_tokens = max_tokens
        self.tool_timeout_s = tool_timeout_s

    def run(self, query: str) -> AgentResult:
        from tinyagent.errors import BackendError, EXEC_TERMINAL
        from tinyagent.executor import execute
        from tinyagent.parser import parse
        from tinyagent.prompt import STOP_SEQUENCES, render_prompt, render_repair_prompt
        from tinyagent.types import AgentResult, ParseFailure, ParsedAction, ParsedFinal

        original = render_prompt(query, self.registry.all())
        current = original
        outputs: list[str] = []
        last_tool_result: ToolResult | None = None

        for attempt in range(1 + self.max_repairs):
            calls = attempt + 1
            try:
                completion = self.backend.complete(
                    current, stop=STOP_SEQUENCES, max_tokens=self.max_tokens
                )
            except BackendError as error:
                return AgentResult(
                    "error", f"{error.code}: {error.detail}", "none",
                    last_tool_result, calls, tuple(outputs), error.code,
                )

            outputs.append(completion.text)
            outcome = parse(completion.text)
            if isinstance(outcome, ParsedAction):
                result = execute(
                    outcome, self.registry, timeout_s=self.tool_timeout_s
                )
                last_tool_result = result
                if result.ok:
                    return AgentResult(
                        "tool", result.output or "", "tool", result,
                        calls, tuple(outputs), None,
                    )
                code = result.error_code or "E_TOOL_EXCEPTION"
                if code in EXEC_TERMINAL:
                    return AgentResult(
                        "error", f"{code}: {result.error_detail or ''}".rstrip(": "),
                        "none", result, calls, tuple(outputs), code,
                    )
                detail = result.error_detail or ""
            elif isinstance(outcome, ParsedFinal):
                if not self.require_tool:
                    return AgentResult(
                        "final", outcome.text, "model", last_tool_result,
                        calls, tuple(outputs), None,
                    )
                code, detail = "E_FINAL_NOT_ALLOWED", ""
            elif isinstance(outcome, ParseFailure):
                code, detail = outcome.code, ""
            else:
                raise TypeError("parser returned an invalid outcome")

            if attempt == self.max_repairs:
                return AgentResult(
                    "error", f"retry exhausted after {code}", "none",
                    last_tool_result, calls, tuple(outputs), "E_RETRY_EXHAUSTED",
                )
            current = render_repair_prompt(
                original, completion.text, code, detail
            )

        raise AssertionError("unreachable bounded loop exit")
