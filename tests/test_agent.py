"""R4 orchestration tests against signature-compatible collaborator stubs."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from slmrouter.agent import SLMRouter


@dataclasses.dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    backend: str = "test"


@dataclasses.dataclass(frozen=True)
class AgentResult:
    status: str
    text: str
    provenance: str
    tool_result: object
    model_calls: int
    raw_outputs: tuple[str, ...]
    error_code: str | None


@dataclasses.dataclass(frozen=True)
class ParsedAction:
    tool: str
    args: dict[str, str]
    raw: str


@dataclasses.dataclass(frozen=True)
class ParsedFinal:
    text: str
    raw: str


@dataclasses.dataclass(frozen=True)
class ParseFailure:
    code: str
    raw: str


@dataclasses.dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    args: dict[str, int]
    output: str | None
    error_code: str | None
    error_detail: str | None
    duration_ms: float = 0.0


class BackendError(Exception):
    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail


class Registry:
    def all(self):
        return ("add", "boom")


class ScriptBackend:
    name = "test"

    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = []

    def complete(self, prompt, *, stop, max_tokens):
        self.prompts.append(prompt)
        value = self.outputs[len(self.prompts) - 1]
        if isinstance(value, Exception):
            raise value
        return Completion(value)


def _module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


def _parse(text):
    if text.startswith("FINAL: "):
        return ParsedFinal(text[7:], text)
    if text.startswith("ACTION: add("):
        return ParsedAction("add", {"a": "2", "b": "3"}, text)
    if text == "ACTION: boom()":
        return ParsedAction("boom", {}, text)
    if text.startswith("ACTION: add_numbers("):
        return ParsedAction("add_numbers", {}, text)
    return ParseFailure("E_NO_PREFIX", text)


def _execute(action, registry, *, timeout_s=10.0, max_output_chars=4000):
    if action.tool == "add":
        return ToolResult(True, "add", {"a": 2, "b": 3}, "5", None, None)
    if action.tool == "boom":
        return ToolResult(False, "boom", {}, None, "E_TOOL_EXCEPTION", "fixture boom")
    return ToolResult(False, action.tool, {}, None, "E_UNKNOWN_TOOL", action.tool)


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.modules = {
            "slmrouter.registry": _module("slmrouter.registry", default_registry=Registry()),
            "slmrouter.types": _module(
                "slmrouter.types", Backend=object, Completion=Completion, AgentResult=AgentResult,
                ParsedAction=ParsedAction, ParsedFinal=ParsedFinal,
                ParseFailure=ParseFailure, ToolResult=ToolResult,
            ),
            "slmrouter.errors": _module(
                "slmrouter.errors", BackendError=BackendError,
                EXEC_TERMINAL=frozenset({"E_TOOL_EXCEPTION", "E_TOOL_TIMEOUT"}),
            ),
            "slmrouter.prompt": _module(
                "slmrouter.prompt", STOP_SEQUENCES=("\n",),
                estimate_tokens=lambda text: (len(text) + 3) // 4,
                render_prompt=lambda query, tools: f"ORIGINAL {query}",
                render_repair_prompt=lambda original, previous, code, detail:
                    f"{original} repair {code} {previous[:200]}",
            ),
            "slmrouter.parser": _module("slmrouter.parser", parse=_parse),
            "slmrouter.executor": _module("slmrouter.executor", execute=_execute),
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_happy_path_and_final(self):
        tool = SLMRouter(ScriptBackend(['ACTION: add(a="2", b="3")'])).run("add")
        self.assertEqual((tool.status, tool.text, tool.provenance, tool.model_calls),
                         ("tool", "5", "tool", 1))
        final = SLMRouter(ScriptBackend(["FINAL: Paris"])).run("capital")
        self.assertEqual((final.status, final.provenance), ("final", "model"))

    def test_repair_and_original_prompt(self):
        backend = ScriptBackend(["ACTION: add_numbers(a=2,b=3)", 'ACTION: add(a="2", b="3")'])
        result = SLMRouter(backend).run("add")
        self.assertEqual((result.status, result.model_calls), ("tool", 2))
        self.assertTrue(backend.prompts[1].startswith(backend.prompts[0]))
        self.assertEqual(len(result.raw_outputs), 2)

    def test_ceiling(self):
        for repairs in (0, 1, 2):
            with self.subTest(repairs=repairs):
                backend = ScriptBackend(["garbage"] * (repairs + 1))
                result = SLMRouter(backend, max_repairs=repairs).run("add")
                self.assertEqual(result.error_code, "E_RETRY_EXHAUSTED")
                self.assertEqual(result.model_calls, repairs + 1)
                self.assertIn("E_NO_PREFIX", result.text)

    def test_terminal_failures_do_not_retry(self):
        backend = ScriptBackend([BackendError("E_BACKEND_TIMEOUT", "timed out")])
        result = SLMRouter(backend).run("add")
        self.assertEqual((result.error_code, result.model_calls),
                         ("E_BACKEND_TIMEOUT", 1))
        result = SLMRouter(ScriptBackend(["ACTION: boom()"])).run("boom")
        self.assertEqual((result.error_code, result.model_calls),
                         ("E_TOOL_EXCEPTION", 1))

    def test_final_requires_tool(self):
        result = SLMRouter(ScriptBackend(["FINAL: Paris", "FINAL: Paris"]),
                           require_tool=True).run("capital")
        self.assertEqual(result.error_code, "E_RETRY_EXHAUSTED")
        self.assertEqual(result.model_calls, 2)

    def test_prior_execution_is_retained_on_final(self):
        result = SLMRouter(
            ScriptBackend(["ACTION: add_numbers(a=2,b=3)", "FINAL: unknown"])
        ).run("add")
        self.assertEqual(result.status, "final")
        self.assertIsNotNone(result.tool_result)
        self.assertEqual(result.tool_result.error_code, "E_UNKNOWN_TOOL")

    def test_repair_prompts_do_not_grow(self):
        backend = ScriptBackend(["x" * 1000] * 3)
        result = SLMRouter(backend, max_repairs=2).run("add")
        self.assertEqual(result.model_calls, 3)
        original = backend.prompts[0]
        for prompt in backend.prompts[1:]:
            self.assertTrue(prompt.startswith(original))
            self.assertLessEqual(len(prompt) - len(original), 400)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            SLMRouter(ScriptBackend([]), max_repairs=3)
        with self.assertRaisesRegex(ValueError, "no tools registered"):
            SLMRouter(ScriptBackend([]), registry=_module("empty", all=lambda: ()))

    def test_record_and_replay_exact_prompt(self):
        original_module = sys.modules.get("slmrouter.backends.simulator")

        def restore_module():
            if original_module is None:
                sys.modules.pop("slmrouter.backends.simulator", None)
            else:
                sys.modules["slmrouter.backends.simulator"] = original_module

        self.addCleanup(restore_module)
        from slmrouter.backends.simulator import RecordingBackend, ReplayBackend, SimulatedBackend

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "run.jsonl"
            inner = SimulatedBackend(["FINAL: fixture"])
            recorded = RecordingBackend(inner, fixture)
            original = recorded.complete("prompt", stop=(), max_tokens=96)
            replay = ReplayBackend(fixture)
            duplicate = replay.complete("prompt", stop=(), max_tokens=96)
            self.assertEqual(original.text, duplicate.text)
            with self.assertRaises(KeyError):
                replay.complete("changed prompt", stop=(), max_tokens=96)


class CorpusShapeTest(unittest.TestCase):
    def test_scenarios_and_faults(self):
        from eval.faults import FAULTS
        from eval.scenarios import scenarios

        self.assertGreaterEqual(len(scenarios()), 30)
        self.assertEqual([fault.id for fault in FAULTS],
                         [f"F{number:02d}" for number in range(1, 17)])
        self.assertEqual(len({scenario.query for scenario in scenarios()}),
                         len(scenarios()))

    def test_live_query_count(self):
        from eval.run_eval import _live_queries

        self.assertEqual(len(_live_queries()), 50)


class ReflectionPromptSimulatorContractTest(unittest.TestCase):
    def test_reflection_prompt_and_simulator_contract(self):
        from examples.demo_tools import build_registry
        from slmrouter.backends.simulator import SimulatedBackend
        from slmrouter.prompt import STOP_SEQUENCES, render_prompt
        from slmrouter.types import Completion

        registry = build_registry()
        prompt = render_prompt("What is 2+3?", registry.all())
        self.assertIn("add(a:int, b:int)", prompt)
        backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
        completion = backend.complete(prompt, stop=STOP_SEQUENCES, max_tokens=96)
        self.assertIsInstance(completion, Completion)
        self.assertEqual(backend.prompts, [prompt])


if __name__ == "__main__":
    unittest.main()
