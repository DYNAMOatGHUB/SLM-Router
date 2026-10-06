# TinyAgent

TinyAgent routes one query to one registered Python tool through a local language
model, with at most `1 + max_repairs` model calls. Its architecture and tradeoffs
are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

Aisha's reflection, registry, prompt, and frozen contract modules are integrated.
The transport, parser, and executor modules are owned by the other engineers
and have not yet landed. The CLI and full evaluation require the parser and
executor. The evaluation uses the PRD-04 fixture signatures in
`examples/demo_tools.py`; Aisha's `tests/fixtures_tools.py` has different
`search` and `set_alarm` signatures.

## Quickstart

```python
from examples.demo_tools import build_registry
from tinyagent import TinyAgent
from tinyagent.backends.simulator import SimulatedBackend

backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
agent = TinyAgent(backend, build_registry())
result = agent.run("What is 2+3?")
print(f"{result.text} (provenance: {result.provenance})")
```

After the other modules land, run `python -m tinyagent --backend sim`.
The CLI labels this mode `[SIMULATED]` and prints provenance next to every
answer. A `FINAL:` answer has `provenance=model`; it is not verified.

## Supported runtimes

The planned live backends are Ollama and llama.cpp server. Tested runtime
versions will be added after Deepak's transport handoff and a live run.
Use `python -m tinyagent --backend ollama --model llama3:8b "What is 2+3?"`
or `--backend llamacpp` when those modules are available.

## Evaluation

`python -m eval.run_eval --mode sim` runs the seeded fault corpus without a GPU
once the parser and executor are available. The latest
[evaluation report](eval/report.md) currently marks both simulated and live
runs as not executed. No live accuracy is claimed until a live run is recorded.

Known limits: one tool per query; no conversation memory, tool chaining, or
complex argument types. A valid but wrong tool choice cannot be detected by the
parser. Registered tools must enforce their own authorization and timeouts;
the executor cannot kill a tool thread after timeout.
