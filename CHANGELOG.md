# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-13

First public release. The project was developed under the working name
TinyAgent and renamed to SLMRouter for publication.

### Added

- `@tool` decorator that builds a tool description from a function's
  signature, type hints and docstring.
- Compact single-prompt rendering (about 100 tokens) with query sanitization and
  a non-growing repair prompt.
- `OllamaBackend` and `LlamaCppBackend` using greedy, seeded decoding over the
  standard library HTTP client. `LlamaCppBackend` applies the model's chat
  template by default.
- Quote-aware parser for `ACTION:` / `FINAL:` replies that tolerates common
  small-model formatting errors and rejects ambiguous output.
- Strict argument coercion for `str`, `int`, `float`, `bool` and `Optional`.
- Allowlisted executor with a per-call timeout, output cap and redacted errors.
- `SLMRouter.run` with at most `1 + max_repairs` model calls and an
  `AgentResult` that always reports provenance.
- `slmrouter` command-line tool, simulated, recording and replay backends, a
  fault-injection evaluation harness and recorded live evaluations.

[Unreleased]: https://github.com/DYNAMOatGHUB/SLM-Router/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DYNAMOatGHUB/SLM-Router/releases/tag/v0.1.0
