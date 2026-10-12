"""Run deterministic, live, or replay TinyAgent evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from eval.faults import FAULTS
from eval.scenarios import Scenario, boom_scenario, scenarios


def _result_matches(result: Any, scenario: Scenario) -> bool:
    tool = result.tool_result
    return bool(
        result.status == "tool"
        and tool is not None
        and tool.tool == scenario.tool
        and dict(tool.args) == scenario.args
    )


def _false_execution(result: Any, scenario: Scenario) -> bool:
    tool = result.tool_result
    return bool(
        tool is not None
        and (result.status == "tool" or result.error_code in {"E_TOOL_EXCEPTION", "E_TOOL_TIMEOUT"})
        and (tool.tool != scenario.tool or dict(tool.args) != scenario.args)
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _round_trip(items: tuple[Scenario, ...], registry: Any) -> None:
    from tinyagent.executor import execute
    from tinyagent.parser import parse
    from tinyagent.types import ParsedAction

    for scenario in items:
        parsed = parse(scenario.canonical_output)
        if not isinstance(parsed, ParsedAction):
            raise ValueError(f"simulator invalid: parse {scenario.query}")
        result = execute(parsed, registry)
        if not result.ok or result.tool != scenario.tool or dict(result.args) != scenario.args:
            raise ValueError(f"simulator invalid: execute {scenario.query}")


def _fault_code(text: str, registry: Any) -> str | None:
    from tinyagent.executor import execute
    from tinyagent.parser import parse
    from tinyagent.types import ParsedAction, ParsedFinal, ParseFailure

    outcome = parse(text)
    if isinstance(outcome, ParseFailure):
        return outcome.code
    if isinstance(outcome, ParsedFinal):
        return "E_FINAL_NOT_ALLOWED"
    if isinstance(outcome, ParsedAction):
        result = execute(outcome, registry)
        return result.error_code
    raise TypeError("parser returned an invalid outcome")


def _sim(seed: int, max_repairs: int) -> tuple[dict[str, Any], bool]:
    from tinyagent.demo_tools import build_registry
    from tinyagent.agent import TinyAgent
    from tinyagent.backends.simulator import SimulatedBackend
    from tinyagent.prompt import estimate_tokens, render_prompt, render_tool_line

    registry = build_registry()
    items = scenarios()
    _round_trip(items, registry)
    rng = random.Random(seed)
    cases = []
    overhead = []
    prompt_tokens = []
    false_executions = 0
    recover_passed = recover_total = 0
    max_calls = 0
    failures = []

    for fault in FAULTS:
        eligible = [s for s in items if fault.id != "F14" or s.tool != "get_weather"]
        rng.shuffle(eligible)
        for scenario in eligible:
            actual = boom_scenario() if fault.id == "F16" else scenario
            corrupted = fault.corrupt(scenario)
            observed_code = _fault_code(corrupted, registry)
            if observed_code != fault.expected_code:
                failures.append(
                    f"{fault.id}/{actual.query}: expected {fault.expected_code}, got {observed_code}"
                )
            for variant in ("repair", "repeat") if fault.verdict == "repair" else ("first",):
                script = lambda prompt, index, bad=corrupted, good=actual.canonical_output, v=variant: (
                    bad if index == 0 or v == "repeat" else good
                )
                backend = SimulatedBackend(script)
                agent = TinyAgent(
                    backend, registry, max_repairs=max_repairs,
                    require_tool=True,
                )
                start = time.perf_counter_ns()
                result = agent.run(actual.query)
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                overhead.append(elapsed_ms)
                prompt_tokens.extend(estimate_tokens(prompt) for prompt in backend.prompts)
                max_calls = max(max_calls, result.model_calls)
                false_executions += int(_false_execution(result, actual))
                if fault.verdict == "recover":
                    recover_total += 1
                    valid = result.model_calls == 1 and _result_matches(result, actual)
                    recover_passed += int(valid)
                elif fault.verdict == "reject":
                    valid = (
                        result.status == "error"
                        and result.error_code == fault.expected_code
                        and result.model_calls == 1
                    )
                elif variant == "repair" and max_repairs > 0:
                    valid = (
                        _result_matches(result, actual)
                        and result.model_calls == 2
                    )
                else:
                    valid = (
                        result.status == "error"
                        and result.error_code == "E_RETRY_EXHAUSTED"
                        and result.model_calls == 1 + max_repairs
                        and fault.expected_code in result.text
                    )
                if not valid:
                    failures.append(f"{fault.id}/{actual.query}/{variant}: {result.error_code or result.status}")
                cases.append((fault.id, variant))

    metrics = {
        "mode": "simulated",
        "seed": seed,
        "scenarios": len(items),
        "cases": len(cases),
        "recover_passed": recover_passed,
        "recover_total": recover_total,
        "recover_rate": recover_passed / recover_total if recover_total else None,
        "false_executions": false_executions,
        "max_model_calls": max_calls,
        "model_call_ceiling": 1 + max_repairs,
        "median_framework_overhead_ms": statistics.median(overhead),
        "base_prompt_tokens": estimate_tokens(
            render_prompt("what is the answer to this short test query today", ())
        ),
        "max_tool_line_tokens": max(
            estimate_tokens(render_tool_line(spec)) for spec in registry.all()
        ),
        "median_prompt_tokens": statistics.median(prompt_tokens),
        "max_prompt_tokens": max(prompt_tokens),
        "failures": failures,
    }
    passed = (
        recover_passed == recover_total
        and false_executions == 0
        and max_calls <= 1 + max_repairs
        and not failures
    )
    return metrics, passed


def _live_queries() -> list[dict[str, Any]]:
    path = Path(__file__).with_name("live_queries.jsonl")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _ollama_metadata(host: str, model: str) -> tuple[str, str]:
    """Best-effort runtime version and quantization; "not measured" on any failure."""
    import urllib.request

    def fetch(path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{host}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    version = quantization = "not measured"
    try:
        version = str(fetch("/api/version")["version"])
        quantization = str(fetch("/api/show", {"model": model})["details"]["quantization_level"])
    except Exception:
        pass
    return version, quantization


def _live(backend_name: str, model: str, record: str | None) -> dict[str, Any]:
    from tinyagent.demo_tools import build_registry
    from tinyagent.agent import TinyAgent
    from tinyagent.backends.simulator import RecordingBackend

    if backend_name == "ollama":
        from tinyagent.backends import OllamaBackend

        backend = OllamaBackend(model)
        runtime_version, quantization = _ollama_metadata(backend.host, model)
    elif backend_name == "llamacpp":
        from tinyagent.backends import LlamaCppBackend

        backend = LlamaCppBackend()
        runtime_version = quantization = "not measured"
    else:
        raise ValueError(f"unsupported live backend: {backend_name}")
    if record:
        backend = RecordingBackend(backend, record)
    agent = TinyAgent(backend, build_registry(), require_tool=True)
    queries = _live_queries()
    results = []
    durations = []
    for case in queries:
        start = time.perf_counter()
        result = agent.run(case["query"])
        durations.append((time.perf_counter() - start) * 1000)
        if result.error_code in {"E_BACKEND_UNREACHABLE", "E_BACKEND_TIMEOUT"}:
            return {"live": "not executed", "reason": result.error_code}
        results.append((case, result))
    tool_correct = sum(
        result.tool_result is not None and result.tool_result.tool == case["tool"]
        for case, result in results
    )
    arg_correct = sum(
        result.tool_result is not None
        and result.tool_result.tool == case["tool"]
        and dict(result.tool_result.args) == case["args"]
        for case, result in results
    )
    return {
        "mode": "live",
        "hardware": platform.machine(),
        "os": platform.platform(),
        "model": model,
        "runtime": backend_name,
        "runtime_version": runtime_version,
        "quantization": quantization,
        "queries": len(results),
        "tool_selection_accuracy": tool_correct / len(results) if results else None,
        "arg_accuracy": arg_correct / len(results) if results else None,
        "repair_usage": sum(result.model_calls > 1 for _, result in results),
        "p50_latency_ms": _percentile(durations, 0.5),
        "p95_latency_ms": _percentile(durations, 0.95),
    }


def _replay(fixture: str) -> dict[str, Any]:
    from tinyagent.demo_tools import build_registry
    from tinyagent.agent import TinyAgent
    from tinyagent.backends.simulator import ReplayBackend

    agent = TinyAgent(ReplayBackend(fixture), build_registry(), require_tool=True)
    results = [agent.run(case["query"]) for case in _live_queries()]
    return {
        "mode": "replay",
        "fixture": fixture,
        "queries": len(results),
        "tool_successes": sum(result.status == "tool" for result in results),
        "max_model_calls": max((result.model_calls for result in results), default=0),
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = ["| Metric | Value |", "|---|---|"]
    for key, value in report.items():
        if key != "failures":
            rows.append(f"| {key} | {value} |")
    markdown = path.with_suffix(".md")
    markdown.write_text(
        "# TinyAgent evaluation\n\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sim", "live", "replay"), default="sim")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), default="ollama")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--record")
    parser.add_argument("--fixture")
    parser.add_argument("--out", default="eval/report.json")
    args = parser.parse_args(argv)
    if args.mode == "sim":
        try:
            report, passed = _sim(args.seed, args.max_repairs)
        except ValueError as error:
            if "simulator invalid" in str(error):
                print(str(error), file=sys.stderr)
                return 2
            raise
    elif args.mode == "live":
        report, passed = _live(args.backend, args.model, args.record), True
    else:
        if not args.fixture:
            parser.error("--fixture is required for replay")
        report, passed = _replay(args.fixture), True
    _write_report(report, Path(args.out))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
