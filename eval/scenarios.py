"""Ground-truth scenarios for deterministic routing evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    query: str
    tool: str
    args: dict[str, Any]
    canonical_output: str


def _action(tool: str, args: dict[str, Any]) -> str:
    values = ", ".join(f'{key}="{value}"' for key, value in args.items())
    return f"ACTION: {tool}({values})"


def scenarios() -> tuple[Scenario, ...]:
    items: list[Scenario] = []
    for a, b in (
        (2, 3), (0, 0), (-4, 9), (17, 25), (100, -1),
        (8, 13), (-12, -5), (42, 58), (1, 999), (123, 456),
    ):
        args = {"a": a, "b": b}
        items.append(Scenario(f"Add {a} and {b}", "add", args, _action("add", args)))
    for limit, q in enumerate(
        ("weather", "Python", "agents", "Pune", "solar", "maps",
         "books", "trains", "music", "science"), 1
    ):
        args = {"q": q, "limit": limit}
        items.append(
            Scenario(f"Search {q}, limit {limit}", "search", args, _action("search", args))
        )
    for hour in range(10):
        args = {"hour": hour, "minute": hour + 10, "label": f"event{hour}"}
        items.append(
            Scenario(
                f"Set alarm for {hour}:{hour + 10:02d}",
                "set_alarm", args, _action("set_alarm", args),
            )
        )
    for city in ("Pune", "Delhi", "Mumbai", "Chennai", "Kolkata"):
        args = {"city": city, "unit": "c"}
        items.append(
            Scenario(f"Weather in {city}", "get_weather", args, _action("get_weather", args))
        )
    return tuple(items)


def boom_scenario() -> Scenario:
    return Scenario("Run the boom test", "boom", {}, "ACTION: boom()")
