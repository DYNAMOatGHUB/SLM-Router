# SLMRouter

[![PyPI](https://img.shields.io/pypi/v/slmrouter)](https://pypi.org/project/slmrouter/)
[![Python](https://img.shields.io/pypi/pyversions/slmrouter)](https://pypi.org/project/slmrouter/)
[![CI](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/actions/workflows/ci.yml/badge.svg)](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/LICENSE)

**Reliable tool calling for small local LLMs.** Zero dependencies, bounded model
calls, no `eval`.

```console
$ slmrouter --backend ollama --model llama3:8b "What's the weather in Pune?"
status: tool
provenance: tool
model_calls: 1
text: Pune: 31 c, clear

$ slmrouter --backend ollama --model llama3:8b "Who wrote Hamlet?"
status: final
provenance: model
model_calls: 1
text: William Shakespeare
```

SLMRouter is a zero-dependency Python library (stdlib only, Python >= 3.10) that
lets a 7B/8B local language model, served by Ollama or llama.cpp, reliably call
your Python functions.

Agent frameworks built for frontier cloud models assume the model can follow
long system prompts, emit valid JSON and recover from its own errors. A local
8B model does not, which leads to context exhaustion, JSON parse crashes and
retry loops that never end. SLMRouter instead treats the model as a
**deterministic router**: each query becomes one bounded decision, either call
one registered tool with arguments, or return a final answer. A query uses at
most `1 + max_repairs` model calls (`max_repairs` is 0, 1 or 2), so infinite
loops are structurally impossible. The full design and tradeoffs are in
[ARCHITECTURE.md](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/ARCHITECTURE.md).

## Install

```bash
pip install slmrouter
```

It has no runtime dependencies.

To install the latest code from GitHub instead:

```bash
pip install "git+https://github.com/DYNAMOatGHUB/TinyAgent-SLM.git"
```

For live use, run a local [Ollama](https://ollama.com) or llama.cpp server.

## Quickstart

```python
from slmrouter import SLMRouter, ToolRegistry, tool
from slmrouter.backends import OllamaBackend

registry = ToolRegistry()

@tool(registry=registry)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = SLMRouter(OllamaBackend("mistral:latest"), registry)
result = agent.run("What is 2+3?")
print(f"{result.text} (provenance: {result.provenance})")  # 5 (provenance: tool)
```

Without a model, use the scripted backend:
`python examples/quickstart.py --backend sim`.

### Command line

```bash
slmrouter --backend sim                                   # scripted, prints [SIMULATED]
slmrouter --backend ollama --model mistral:latest "What's the weather in Pune?"
slmrouter --backend llamacpp --host http://127.0.0.1:8080 "Add 4 and 9" --verbose
```

`python -m slmrouter` is equivalent. The CLI routes to the demo tools in
`slmrouter/demo_tools.py`; exit code is 1 when `status` is `error`.

## Reading results

`agent.run()` returns an `AgentResult`. Always show `provenance` next to the
answer:

| `status` | `provenance` | `text` | Meaning |
|---|---|---|---|
| `tool` | `tool` | Tool output, unmodified | A registered tool ran with validated arguments |
| `final` | `model` | The model's own answer | No tool was used. **This text is not verified.** Pass `require_tool=True` to disallow it |
| `error` | `none` | `CODE: detail` | See `error_code`; `run()` does not raise for model, backend or tool failures |

`run()` raises only `ValueError`, for an empty query or invalid configuration.

Error codes: parse (`E_EMPTY`, `E_NO_PREFIX`, `E_BAD_SYNTAX`) and argument
(`E_UNKNOWN_TOOL`, `E_UNEXPECTED_ARG`, `E_MISSING_ARG`, `E_TYPE_COERCION`)
failures trigger a repair prompt; if repairs run out the result is
`E_RETRY_EXHAUSTED`. Backend failures (`E_BACKEND_*`) and tool failures
(`E_TOOL_EXCEPTION`, `E_TOOL_TIMEOUT`) are never retried, because tools may
have side effects.

## Supported runtimes

| Runtime | Endpoint | Tested |
|---|---|---|
| Ollama | `POST /api/generate` | Ollama 0.34.0 with `llama3:8b` (Q4_0) and `mistral:latest` (7.2B, Q4_K_M), Linux x86_64, NVIDIA RTX 5070 |
| llama.cpp server | `POST /v1/chat/completions` (default) or `POST /completion` | llama.cpp b11565 (CPU build), same mistral 7B Q4_K_M weights, Linux x86_64 |

Both backends use greedy, seeded decoding (`temperature 0`, `top_k 1`). They
warn when the host is not loopback, because prompts would leave the machine.

Both backends apply the model's chat template, which instruction-tuned models
need. Ollama does this by default; `LlamaCppBackend` does it by using the
server's chat endpoint. Use `LlamaCppBackend(chat_template=False)` only for
base (non-instruct) models: it sends the raw prompt to `/completion`. With
mistral-7b-instruct, raw prompts chose the right tool for only 27 of 50
benchmark queries, against 50 of 50 with the template.

## Writing tools

- Parameters may be `str`, `int`, `float`, `bool` or `Optional[...]` of those.
  `*args`, `**kwargs`, positional-only and complex types are rejected at
  decoration time.
- The first docstring line (max 80 chars) becomes the tool description the
  model sees. Clear descriptions are the main lever on tool-selection accuracy.
- Arguments are coerced strictly: `"2.5"` is not an `int`, `"nan"` is not a
  `float`, and booleans accept only `true/false/yes/no/1/0/on/off`.
- Tool output is returned as data. It is never parsed again or executed.

## Security model

- Model text never becomes code: no `eval`, `exec`, `compile`, `pickle` or
  `importlib` in the package (enforced by an AST gate in CI).
- Only registered tools can run, by exact name. There is no fuzzy matching.
- Ambiguous output (unterminated quotes, duplicate arguments, non-integral
  integers) is rejected, never guessed.
- **Prompt injection:** a user query can still persuade the model to call any
  registered tool. Tools that change state must enforce their own authorization.
- **Timeouts:** tools run in a daemon thread with a timeout (default 10 s).
  Python cannot kill threads, so a timed-out tool keeps running in the
  background. Tools that call external systems must enforce their own timeouts.
- Error details are single-line and capped; prompts are redacted from backend
  errors and file paths from tool exceptions.

## Evaluation

| Run | Command | Result |
|---|---|---|
| Simulated (no model, CI) | `python -m eval.run_eval --mode sim` | 795 fault-injection cases: 280/280 recoverable faults recovered on the first call, **0 false executions**, max 2 model calls (ceiling 2), median framework overhead 0.05 ms — [report](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/eval/report.md) |
| Live | `python -m eval.run_eval --mode live --backend ollama --model llama3:8b --record eval/fixtures/live-llama3.jsonl --out eval/report_live_llama3.json` | See the table below |
| Replay | `python -m eval.run_eval --mode replay --fixture eval/fixtures/live-llama3.jsonl` | Re-runs a recorded live session exactly; every recording in `eval/fixtures/` is also replayed in CI by `tests/test_replay.py` |

Live results on the 50-query benchmark (`eval/live_queries.jsonl`), measured on
Linux x86_64 with an NVIDIA RTX 5070 on 2026-10-12:

| Runtime | Model | Tool selection | Arguments | Repairs | p50 / p95 latency | Report |
|---|---|---|---|---|---|---|
| Ollama 0.34.0 (GPU) | `llama3:8b` Q4_0 | **50/50** | **50/50** | 0 | 109 / 156 ms | [report](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/eval/report_live_llama3.md) |
| Ollama 0.34.0 (GPU) | `mistral:latest` 7.2B Q4_K_M | **50/50** | **50/50** | 0 | 130 / 184 ms | [report](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/eval/report_live.md) |
| llama.cpp b11565 (CPU only) | same mistral 7B Q4_K_M weights | **50/50** | **50/50** | 0 | 1378 / 1634 ms | [report](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/eval/report_live_llamacpp.md) |

These are measurements on one machine, not guarantees. Other models,
quantizations or hardware can differ, and the benchmark covers only the four
demo tools.

## Development

```bash
python -m unittest discover -s tests -t .   # unit, integration, gate and replay tests
python -m eval.run_eval --mode sim          # simulated evaluation (exit 1 on any NFR failure)
python -m bench.bench_parser                # parser p99 latency (NFR-8)
```

CI runs all three on Python 3.10, 3.12 and 3.13, plus a clean-install smoke test.

## Known limitations

- One tool per query; no conversation memory, tool chaining, async API or
  streaming.
- No complex argument types (`list`, `dict`, nested objects).
- A valid but wrong tool choice cannot be detected by the parser; it is
  returned with `provenance="tool"`. The live evaluation measures how often this
  happens.
- Temperature 0 is not bit-identical across hardware and runtime versions; use
  record/replay for reproducible tests.
- No OS-level sandboxing: SLMRouter controls *which* registered function runs
  and *with what* validated arguments, not what the function does.

## Contributing

Bug reports, questions and pull requests are welcome. See
[CONTRIBUTING.md](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/CONTRIBUTING.md)
for how to set up a development environment and what a pull request needs.
Report security issues privately as described in
[SECURITY.md](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/SECURITY.md),
not in a public issue.

## License

SLMRouter is released under the
[Apache License 2.0](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/LICENSE).
Copyright 2026 Dhyanesh Mahendiran. See
[NOTICE](https://github.com/DYNAMOatGHUB/TinyAgent-SLM/blob/main/NOTICE).

The SLMRouter name is not licensed for use by derivative projects
(Apache-2.0, Section 6).
