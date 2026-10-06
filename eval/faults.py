"""Forward-generated faults from PRD-03 Section 7.

The verdict table requires co-review by the parser owner before release.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from eval.scenarios import Scenario


@dataclass(frozen=True)
class Fault:
    id: str
    verdict: str
    expected_code: str | None
    corrupt: Callable[[Scenario], str]


def _replace_first_arg(s: Scenario, name: str, value: str | None = None) -> str:
    first = next(iter(s.args))
    old = f'{first}="{s.args[first]}"'
    if value is None:
        return s.canonical_output.replace(old, f'{name}="{s.args[first]}"', 1)
    return s.canonical_output.replace(old, f'{name}="{value}"', 1)


def _positional(s: Scenario) -> str:
    values = ", ".join(f'"{value}"' for value in s.args.values())
    return f"ACTION: {s.tool}({values})"


def _drop_required(s: Scenario) -> str:
    first = next(iter(s.args))
    parts = [f'{key}="{value}"' for key, value in s.args.items() if key != first]
    return f'ACTION: {s.tool}({", ".join(parts)})'


def _bad_int(s: Scenario) -> str:
    name = next(key for key in s.args if key in {"a", "b", "limit", "hour", "minute"})
    return s.canonical_output.replace(f'{name}="{s.args[name]}"', f'{name}="two"', 1)


FAULTS = (
    Fault("F01", "recover", None, lambda s: f"```\n{s.canonical_output}\n```"),
    Fault("F02", "recover", None, lambda s: f"Here is the answer:\n{s.canonical_output}"),
    Fault("F03", "recover", None, lambda s: f"{s.canonical_output}\nThat is all."),
    Fault("F04", "recover", None, lambda s: s.canonical_output.replace("ACTION:", "action:", 1)),
    Fault("F05", "recover", None, lambda s: s.canonical_output.replace('"', "'")),
    Fault("F06", "recover", None, lambda s: re.sub(r'="([+-]?\d+)"', r'=\1', s.canonical_output)),
    Fault("F07", "recover", None, _positional),
    Fault("F08", "recover", None, lambda s: s.canonical_output[:-1] + ",)"),
    Fault("F09", "repair", "E_NO_PREFIX", lambda s: json.dumps({"tool": s.tool, "args": s.args})),
    Fault("F10", "repair", "E_BAD_SYNTAX", lambda s: s.canonical_output[:-1]),
    Fault("F11", "repair", "E_UNKNOWN_TOOL", lambda s: s.canonical_output.replace(
        f"ACTION: {s.tool}(", f"ACTION: {s.tool}_numbers(", 1)),
    Fault("F12", "repair", "E_UNEXPECTED_ARG", lambda s: _replace_first_arg(s, "x")),
    Fault("F13", "repair", "E_MISSING_ARG", _drop_required),
    Fault("F14", "repair", "E_TYPE_COERCION", _bad_int),
    Fault("F15", "repair", "E_EMPTY", lambda s: ""),
    Fault("F16", "reject", "E_TOOL_EXCEPTION", lambda s: "ACTION: boom()"),
)
