"""PRD-04 acceptance tests against the real parser, executor and prompt modules."""

from __future__ import annotations

import threading
import unittest

from tinyagent.demo_tools import build_registry
from tinyagent import TinyAgent
from tinyagent.backends.simulator import SimulatedBackend
from tinyagent.errors import BackendError
from tinyagent.types import Completion


class FailingBackend:
    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, *, stop, max_tokens) -> Completion:
        self.calls += 1
        raise BackendError("E_BACKEND_TIMEOUT", "no response within 60s")


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def run_agent(self, script, **kwargs):
        backend = SimulatedBackend(script)
        return TinyAgent(backend, self.registry, **kwargs).run("What is 2+3?"), backend

    def test_happy_path(self) -> None:  # A4-1
        result, _ = self.run_agent(['ACTION: add(a="2", b="3")'])
        self.assertEqual(
            (result.status, result.text, result.provenance, result.model_calls),
            ("tool", "5", "tool", 1),
        )
        self.assertEqual(dict(result.tool_result.args), {"a": 2, "b": 3})

    def test_final_allowed(self) -> None:  # A4-2
        result, _ = self.run_agent(["FINAL: Paris"])
        self.assertEqual((result.status, result.provenance, result.text),
                         ("final", "model", "Paris"))

    def test_final_with_require_tool(self) -> None:  # A4-3
        result, _ = self.run_agent(["FINAL: Paris", "FINAL: Paris"], require_tool=True)
        self.assertEqual((result.error_code, result.model_calls), ("E_RETRY_EXHAUSTED", 2))
        self.assertIn("E_FINAL_NOT_ALLOWED", result.text)

    def test_repair_success(self) -> None:  # A4-4
        result, backend = self.run_agent(
            ["ACTION: add_numbers(a=2,b=3)", 'ACTION: add(a="2", b="3")']
        )
        self.assertEqual((result.status, result.text, result.model_calls), ("tool", "5", 2))
        self.assertTrue(backend.prompts[1].startswith(backend.prompts[0]))
        self.assertIn('unknown tool "add_numbers"', backend.prompts[1])

    def test_ceiling(self) -> None:  # A4-5, S4-1
        for repairs in (0, 1, 2):
            with self.subTest(max_repairs=repairs):
                result, backend = self.run_agent(
                    lambda prompt, index: "garbage", max_repairs=repairs
                )
                self.assertEqual(result.model_calls, 1 + repairs)
                self.assertEqual(len(backend.prompts), 1 + repairs)
                self.assertEqual(result.error_code, "E_RETRY_EXHAUSTED")

    def test_backend_error_not_retried(self) -> None:  # A4-6
        backend = FailingBackend()
        result = TinyAgent(backend, self.registry, max_repairs=2).run("add")
        self.assertEqual((result.status, result.provenance), ("error", "none"))
        self.assertEqual((result.error_code, result.model_calls, backend.calls),
                         ("E_BACKEND_TIMEOUT", 1, 1))

    def test_tool_exception_not_retried(self) -> None:  # A4-7
        result, backend = self.run_agent(["ACTION: boom()", "unused"], max_repairs=2)
        self.assertEqual((result.error_code, result.model_calls), ("E_TOOL_EXCEPTION", 1))
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(result.tool_result.error_detail, "RuntimeError: fixture boom")

    def test_prompt_does_not_grow(self) -> None:  # A4-8
        long_garbage = "ACTION: add_numbers(" + "x" * 1000
        _, backend = self.run_agent(lambda prompt, index: long_garbage, max_repairs=2)
        original = backend.prompts[0]
        for prompt in backend.prompts[1:]:
            self.assertTrue(prompt.startswith(original))
            self.assertLessEqual(len(prompt) - len(original), 400)

    def test_empty_query_is_programmer_error(self) -> None:  # R4-14
        with self.assertRaises(ValueError):
            TinyAgent(SimulatedBackend([]), self.registry).run("   ")

    def test_shared_agent_is_thread_safe(self) -> None:  # R4-17
        backend = SimulatedBackend(lambda prompt, index: 'ACTION: add(a="2", b="3")')
        agent = TinyAgent(backend, self.registry)
        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(25):
                result = agent.run("What is 2+3?")
                with lock:
                    results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 200)
        self.assertTrue(all(r.text == "5" and r.model_calls == 1 for r in results))


if __name__ == "__main__":
    unittest.main()
