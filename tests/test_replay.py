"""Regression test over recorded live model outputs (no model or network needed).

The fixture was recorded with `python -m eval.run_eval --mode live --record ...`
and contains only queries from eval/live_queries.jsonl (R4-37).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from slmrouter import SLMRouter
from slmrouter.backends.simulator import ReplayBackend
from slmrouter.demo_tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "eval" / "live_queries.jsonl"
FIXTURES = sorted((ROOT / "eval" / "fixtures").glob("live-*.jsonl"))


class ReplayRegressionTest(unittest.TestCase):
    def test_recorded_live_runs_still_route_correctly(self) -> None:
        self.assertTrue(FIXTURES, "no recorded live fixtures found")
        cases = [json.loads(line) for line in QUERIES.read_text("utf-8").splitlines() if line]
        for fixture in FIXTURES:
            agent = SLMRouter(ReplayBackend(fixture), build_registry(), require_tool=True)
            for case in cases:
                with self.subTest(fixture=fixture.name, query=case["query"]):
                    result = agent.run(case["query"])
                    self.assertEqual(result.status, "tool")
                    self.assertEqual(result.tool_result.tool, case["tool"])
                    self.assertEqual(dict(result.tool_result.args), case["args"])


if __name__ == "__main__":
    unittest.main()
