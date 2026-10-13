# Contributing to SLMRouter

Thank you for helping. Bug reports, questions, documentation fixes and code are
all welcome.

## Reporting a bug or asking for a feature

Open an issue using one of the templates. For a bug, include the SLMRouter
version, Python version, runtime (Ollama or llama.cpp, with its version), the
model, and the smallest code that reproduces the problem. If it is a routing
problem, run the CLI with `--verbose` and paste the printed prompt and raw model
output.

Security problems must not be reported in public issues. See
[SECURITY.md](SECURITY.md).

## Development setup

SLMRouter has no runtime dependencies and no development dependencies beyond
Python 3.10 or newer.

```bash
git clone https://github.com/DYNAMOatGHUB/SLM-Router.git
cd SLM-Router
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run the same checks CI runs:

```bash
python -m unittest discover -s tests -t .   # all tests
python -m eval.run_eval --mode sim          # simulated evaluation, exits 1 on failure
python -m bench.bench_parser                # parser latency benchmark
```

No model is needed for any of these. To try a live model, install
[Ollama](https://ollama.com), run `ollama pull llama3:8b`, then
`python -m slmrouter --backend ollama "What is 2+3?" --verbose`.

## Rules every change must follow

These are design guarantees, and CI enforces most of them. A pull request that
breaks one will not be merged.

- **No runtime dependencies.** Standard library only (`tests/test_no_deps.py`).
- **Model text never becomes code.** No `eval`, `exec`, `compile`,
  `__import__`, `ast.literal_eval`, `pickle` or `importlib` in `slmrouter/`
  (`tests/test_static_gate.py`).
- **Bounded model calls.** A query uses at most `1 + max_repairs` model calls.
- **Fail closed.** Ambiguous model output is rejected with an error code, never
  guessed.
- **Tools are never retried** after they raise or time out.
- The public signatures and the contracts in `slmrouter/types.py` and
  `slmrouter/errors.py` are listed in Section 8 of
  [ARCHITECTURE.md](ARCHITECTURE.md). Open an issue to discuss a change to them
  before writing code.
- Changes to the prompt template need a live evaluation run showing no
  regression (`python -m eval.run_eval --mode live ...`).

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep the change focused on one thing, and match the style of the
   surrounding code.
3. Add or update tests. Bug fixes should include a test that fails without the
   fix.
4. Update `README.md` if behaviour visible to users changes, and add a line
   under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
5. Make sure all three checks above pass, then open the pull request and fill
   in the template.

## Releasing (maintainers)

1. Move the "Unreleased" entries in `CHANGELOG.md` under the new version.
2. Set the same version in `pyproject.toml` and `slmrouter/__init__.py`.
3. Merge to `main`, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.

The `Release` workflow checks that the tag matches the version, runs the tests,
publishes to PyPI and creates a GitHub release.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), as described in Section 5 of that license.
