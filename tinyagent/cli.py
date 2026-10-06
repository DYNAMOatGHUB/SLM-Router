"""Command-line demo entry point."""

from __future__ import annotations

import argparse


class _TracingBackend:
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.prompts: list[str] = []

    def complete(self, prompt, *, stop, max_tokens):
        self.prompts.append(prompt)
        return self.inner.complete(prompt, stop=stop, max_tokens=max_tokens)


def main(argv: list[str] | None = None) -> int:
    from tinyagent import ToolRegistry, tiny_tool
    from tinyagent.agent import TinyAgent
    from tinyagent.backends.simulator import SimulatedBackend

    parser = argparse.ArgumentParser(prog="python -m tinyagent")
    parser.add_argument("query", nargs="?", default="What is 2+3?")
    parser.add_argument("--backend", choices=("sim", "ollama", "llamacpp"), default="sim")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.backend == "sim":
        backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
        print("[SIMULATED]")
    elif args.backend == "ollama":
        from tinyagent.backends import OllamaBackend

        backend = OllamaBackend(args.model)
    else:
        from tinyagent.backends import LlamaCppBackend

        backend = LlamaCppBackend()

    registry = ToolRegistry()

    @tiny_tool(registry=registry)
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    backend = _TracingBackend(backend)
    result = TinyAgent(backend, registry).run(args.query)
    print(f"status: {result.status}")
    print(f"provenance: {result.provenance}")
    print(f"model_calls: {result.model_calls}")
    print(f"text: {result.text}")
    if args.verbose:
        from tinyagent.prompt import estimate_tokens

        for index, (prompt, raw) in enumerate(
            zip(backend.prompts, result.raw_outputs), 1
        ):
            print(f"prompt[{index}] ({estimate_tokens(prompt)} est. tokens):\n{prompt}")
            print(f"raw[{index}] ({estimate_tokens(raw)} est. tokens): {raw}")
    return 0 if result.status != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
