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
        attempt = 0

        def finish(status, text, provenance, error_code=None) -> AgentResult:
            # Reads the loop state (attempt, outputs, last_tool_result) at call time.
            return AgentResult(
                status, text, provenance, last_tool_result,
                attempt + 1, tuple(outputs), error_code,
            )

        for attempt in range(1 + self.max_repairs):
            try:
                completion = self.backend.complete(
                    current, stop=STOP_SEQUENCES, max_tokens=self.max_tokens
                )
            except BackendError as error:
                return finish("error", f"{error.code}: {error.detail}", "none", error.code)

            outputs.append(completion.text)
            outcome = parse(completion.text)
            if isinstance(outcome, ParsedAction):
                result = execute(
                    outcome, self.registry, timeout_s=self.tool_timeout_s
                )
                last_tool_result = result
                if result.ok:
                    return finish("tool", result.output or "", "tool")
                code = result.error_code or "E_TOOL_EXCEPTION"
                detail = result.error_detail or ""
                if code in EXEC_TERMINAL:
                    return finish("error", f"{code}: {detail}".rstrip(": "), "none", code)
            elif isinstance(outcome, ParsedFinal):
                if not self.require_tool:
                    return finish("final", outcome.text, "model")
                code, detail = "E_FINAL_NOT_ALLOWED", ""
            elif isinstance(outcome, ParseFailure):
                code, detail = outcome.code, ""
            else:
                raise TypeError("parser returned an invalid outcome")

            if attempt == self.max_repairs:
                return finish(
                    "error", f"retry exhausted after {code}", "none", "E_RETRY_EXHAUSTED"
                )
            current = render_repair_prompt(
                original, completion.text, code, detail
            )

        raise AssertionError("unreachable bounded loop exit")
