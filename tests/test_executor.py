"""PRD-03 executor tests (A3-3, A3-4) and fault verdict table (A3-9)."""

from __future__ import annotations

import time
import unittest
from typing import Optional

from eval.faults import FAULTS
from eval.scenarios import boom_scenario, scenarios
from tinyagent.demo_tools import build_registry
from tinyagent.executor import TRUNCATION_MARKER, execute
from tinyagent.parser import parse
from tinyagent.reflection import tiny_tool
from tinyagent.registry import ToolRegistry
from tinyagent.types import ParsedAction


def _action(tool: str, **args: str) -> ParsedAction:
    return ParsedAction(tool, args, "")


class CountingRegistry:
    """A registry of tools that record how often they are called."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.registry = ToolRegistry()
        counted = self._counted

        @tiny_tool(registry=self.registry)
        def add(a: int, b: int) -> int:
            """Add two integers."""
            counted("add")
            return a + b

        @tiny_tool(registry=self.registry)
        def greet(name: str, punctuation: str = "!", times: Optional[int] = None) -> str:
            """Greet someone."""
            counted("greet")
            return f"hi {name}{punctuation}" * (times or 1)

        @tiny_tool(registry=self.registry)
        def nothing() -> None:
            """Return nothing."""
            counted("nothing")

        @tiny_tool(registry=self.registry)
        def boom() -> None:
            """Always fails."""
            counted("boom")
            raise RuntimeError("bad\nthing at /home/user/secret/file.py")

        @tiny_tool(registry=self.registry)
        def big() -> str:
            """Large output."""
            counted("big")
            return "x" * 5000

        @tiny_tool(registry=self.registry)
        def slow() -> str:
            """Sleeps."""
            counted("slow")
            time.sleep(2)
            return "late"

        @tiny_tool(registry=self.registry)
        def bad_str() -> object:
            """Returns an object whose str() raises."""
            counted("bad_str")

            class Unprintable:
                def __str__(self) -> str:
                    raise ValueError("cannot print")

            return Unprintable()

    def _counted(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1


class ExecutorStepsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = CountingRegistry()

    def run_action(self, action: ParsedAction, **kwargs):
        return execute(action, self.tools.registry, **kwargs)

    def assert_failure(self, result, code: str, detail: str, tool: str) -> None:
        self.assertFalse(result.ok)
        self.assertEqual((result.error_code, result.error_detail), (code, detail))
        self.assertIsNone(result.output)
        self.assertEqual(dict(result.args), {})  # R3-58
        self.assertEqual(self.tools.calls.get(tool, 0), 0)

    def test_success_with_coercion(self) -> None:
        result = self.run_action(_action("add", a="2", b=" 3 "))
        self.assertTrue(result.ok)
        self.assertEqual((result.output, dict(result.args)), ("5", {"a": 2, "b": 3}))
        self.assertEqual(self.tools.calls["add"], 1)
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_positional_and_defaults(self) -> None:  # R3-51, R3-54
        result = self.run_action(ParsedAction("greet", {"#0": "Ann"}, ""))
        self.assertEqual(result.output, "hi Ann!")
        self.assertEqual(dict(result.args), {"name": "Ann", "punctuation": "!", "times": None})

    def test_optional_none(self) -> None:
        result = self.run_action(_action("greet", name="Ann", times="null"))
        self.assertEqual(result.output, "hi Ann!")

    def test_unknown_tool(self) -> None:  # R3-50, S3-2
        for name in ("subtract", "ADD", "Add", "add_numbers"):
            with self.subTest(name=name):
                result = self.run_action(_action(name, a="1", b="2"))
                self.assert_failure(result, "E_UNKNOWN_TOOL", name, "add")

    def test_too_many_positionals(self) -> None:  # R3-51
        result = self.run_action(ParsedAction("add", {"#0": "1", "#1": "2", "#2": "3"}, ""))
        self.assert_failure(result, "E_UNEXPECTED_ARG", "#2", "add")

    def test_positional_collides_with_keyword(self) -> None:  # R3-51
        result = self.run_action(ParsedAction("add", {"#0": "1", "a": "2"}, ""))
        self.assert_failure(result, "E_UNEXPECTED_ARG", "a", "add")

    def test_unknown_keyword_in_input_order(self) -> None:  # R3-52
        result = self.run_action(_action("add", a="1", z="2", y="3"))
        self.assert_failure(result, "E_UNEXPECTED_ARG", "z", "add")

    def test_missing_in_signature_order(self) -> None:  # R3-53
        result = self.run_action(_action("add"))
        self.assert_failure(result, "E_MISSING_ARG", "a", "add")

    def test_type_coercion(self) -> None:  # R3-54
        result = self.run_action(_action("add", a="1", b="two"))
        self.assert_failure(result, "E_TYPE_COERCION", "b", "add")

    def test_tool_exception_is_contained(self) -> None:  # R3-56, R3-62
        result = self.run_action(_action("boom"))
        self.assertEqual(result.error_code, "E_TOOL_EXCEPTION")
        self.assertEqual(result.error_detail, "RuntimeError: bad thing at [path]")
        self.assertEqual(self.tools.calls["boom"], 1)

    def test_str_failure_is_exception(self) -> None:  # R3-61
        result = self.run_action(_action("bad_str"))
        self.assertEqual(result.error_code, "E_TOOL_EXCEPTION")
        self.assertEqual(result.error_detail, "ValueError: cannot print")
        self.assertEqual(self.tools.calls["bad_str"], 1)

    def test_none_and_truncation(self) -> None:  # R3-57
        self.assertEqual(self.run_action(_action("nothing")).output, "")
        result = self.run_action(_action("big"), max_output_chars=100)
        self.assertEqual(result.output, "x" * 100 + TRUNCATION_MARKER)

    def test_timeout(self) -> None:  # A3-4
        started = time.perf_counter()
        result = self.run_action(_action("slow"), timeout_s=0.2)
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertEqual((result.error_code, result.error_detail),
                         ("E_TOOL_TIMEOUT", "exceeded 0.2s"))
        self.assertEqual(self.tools.calls["slow"], 1)

    def test_invalid_limits_are_programmer_errors(self) -> None:
        for kwargs in ({"timeout_s": 0}, {"timeout_s": float("nan")}, {"max_output_chars": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.run_action(_action("nothing"), **kwargs)


class FaultVerdictTableTest(unittest.TestCase):
    """A3-9: every F01-F16 verdict holds through parse + execute directly."""

    def test_fault_table(self) -> None:
        registry = build_registry()
        for fault in FAULTS:
            for scenario in scenarios():
                if fault.id == "F14" and scenario.tool == "get_weather":
                    continue  # get_weather has no int argument to corrupt.
                expected = boom_scenario() if fault.id == "F16" else scenario
                with self.subTest(fault=fault.id, query=scenario.query):
                    outcome = parse(fault.corrupt(scenario))
                    if not isinstance(outcome, ParsedAction):
                        self.assertEqual(outcome.code, fault.expected_code)
                        continue
                    result = execute(outcome, registry)
                    if fault.verdict == "recover":
                        self.assertTrue(result.ok)
                        self.assertEqual(result.tool, expected.tool)
                        self.assertEqual(dict(result.args), expected.args)
                    else:
                        self.assertEqual(result.error_code, fault.expected_code)


if __name__ == "__main__":
    unittest.main()
