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
    from slmrouter.demo_tools import build_registry
    from slmrouter.agent import SLMRouter
    from slmrouter.backends.simulator import SimulatedBackend

    parser = argparse.ArgumentParser(
        prog="python -m slmrouter",
        description="Route one query to one demo tool through a local model.",
    )
    parser.add_argument("query", nargs="?", default="What is 2+3?")
    parser.add_argument("--backend", choices=("sim", "ollama", "llamacpp"), default="sim")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model tag")
    parser.add_argument("--host", help="Ollama host or llama.cpp base URL")
    parser.add_argument("--timeout", type=float, default=60.0, help="backend timeout (s)")
    parser.add_argument("--max-repairs", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--require-tool", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error("query must not be empty")

    if args.backend == "sim":
        backend = SimulatedBackend(['ACTION: add(a="2", b="3")'])
        print("[SIMULATED]")
    elif args.backend == "ollama":
        from slmrouter.backends import OllamaBackend

        options = {"host": args.host} if args.host else {}
        backend = OllamaBackend(args.model, timeout_s=args.timeout, **options)
    else:
        from slmrouter.backends import LlamaCppBackend

        options = {"base_url": args.host} if args.host else {}
        backend = LlamaCppBackend(timeout_s=args.timeout, **options)

    registry = build_registry()

    backend = _TracingBackend(backend)
    agent = SLMRouter(
        backend, registry, max_repairs=args.max_repairs, require_tool=args.require_tool
    )
    result = agent.run(args.query)
    print(f"status: {result.status}")
    print(f"provenance: {result.provenance}")
    print(f"model_calls: {result.model_calls}")
    print(f"text: {result.text}")
    if result.error_code:
        print(f"error_code: {result.error_code}")
    if args.verbose:
        from slmrouter.prompt import estimate_tokens

        for index, (prompt, raw) in enumerate(
            zip(backend.prompts, result.raw_outputs), 1
        ):
            print(f"prompt[{index}] ({estimate_tokens(prompt)} est. tokens):\n{prompt}")
            print(f"raw[{index}] ({estimate_tokens(raw)} est. tokens): {raw}")
    return 0 if result.status != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
