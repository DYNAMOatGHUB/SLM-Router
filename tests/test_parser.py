"""PRD-03 parser tests: truth table (A3-1), fuzz (A3-5) and ReDoS guard (A3-6)."""

from __future__ import annotations

import random
import string
import time
import unittest

from tests.fixtures_tools import fixture_registry
from tinyagent.executor import execute
from tinyagent.parser import parse
from tinyagent.types import ParsedAction, ParsedFinal, ParseFailure

ADD = ParsedAction("add", {"a": "2", "b": "3"}, "")

# (id, input, expected) where expected is a ParsedAction/ParsedFinal template
# (raw is ignored) or a failure code string.
TRUTH_TABLE = (
    ("P01", "ACTION: add(a=2, b=3)", ADD),
    ("P02", 'ACTION: get_weather(city="Pune")', ParsedAction("get_weather", {"city": "Pune"}, "")),
    ("P03", "```\nACTION: add(a=2, b=3)\n```", ADD),
    ("P04", "Sure! ACTION: add(a=2, b=3)", ADD),
    ("P05", "ACTION: add(a=2, b=3) because it adds.", ADD),
    ("P06", "action : add(a=2,b=3)", ADD),
    ("P07", "ACTION: get_weather(city='Pune')", ParsedAction("get_weather", {"city": "Pune"}, "")),
    ("P08", "ACTION: add(2, 3)", ParsedAction("add", {"#0": "2", "#1": "3"}, "")),
    ("P09", 'ACTION: search(q="a, b (c)")', ParsedAction("search", {"q": "a, b (c)"}, "")),
    ("P10", r'ACTION: say(text="He said \"hi\"")', ParsedAction("say", {"text": 'He said "hi"'}, "")),
    ("P11", "ACTION: add(a=1,)", ParsedAction("add", {"a": "1"}, "")),
    ("P12", "ACTION: ping()", ParsedAction("ping", {}, "")),
    ("P13", "**ACTION:** add(a=1, b=2)", ParsedAction("add", {"a": "1", "b": "2"}, "")),
    ("P14", "FINAL: Paris", ParsedFinal("Paris", "")),
    ("P15", "FINAL:", "E_BAD_SYNTAX"),
    ("P16a", "", "E_EMPTY"),
    ("P16b", "   ", "E_EMPTY"),
    ("P17", '{"tool": "add", "args": {"a": 1}}', "E_NO_PREFIX"),
    ("P18", "ACTION: add(a=2, b=3", "E_BAD_SYNTAX"),
    ("P19", "ACTION: add a=2", "E_BAD_SYNTAX"),
    ("P20", "ACTION: add(a=1, 2)", "E_BAD_SYNTAX"),
    ("P21", "ACTION: add(a=1, a=2)", "E_BAD_SYNTAX"),
    ("P22", "ACTION: add(a=)", "E_BAD_SYNTAX"),
    ("P23", "ACTION: a(x=1)\nACTION: b(y=2)",
     ParsedAction("a", {"x": "1"}, "", ("multiple_actions",))),
    ("P24", 'ACTION: f(x="unterminated)', "E_BAD_SYNTAX"),
    ("P25", 'ACTION: __import__(name="os")', ParsedAction("__import__", {"name": "os"}, "")),
)


class ParserTruthTableTest(unittest.TestCase):
    def test_truth_table(self) -> None:
        for row_id, text, expected in TRUTH_TABLE:
            with self.subTest(row_id):
                outcome = parse(text)
                if isinstance(expected, str):
                    self.assertIsInstance(outcome, ParseFailure)
                    self.assertEqual(outcome.code, expected)
                elif isinstance(expected, ParsedAction):
                    self.assertIsInstance(outcome, ParsedAction)
                    self.assertEqual(outcome.tool, expected.tool)
                    self.assertEqual(dict(outcome.args), dict(expected.args))
                    self.assertEqual(outcome.warnings, expected.warnings)
                else:
                    self.assertIsInstance(outcome, ParsedFinal)
                    self.assertEqual(outcome.text, expected.text)


class ParserRulesTest(unittest.TestCase):
    def test_non_string_input_is_empty(self) -> None:  # R3-01
        for value in (None, 42, b"ACTION: add(a=1)", ["ACTION: f()"]):
            with self.subTest(value=value):
                outcome = parse(value)  # type: ignore[arg-type]
                self.assertEqual(outcome.code, "E_EMPTY")
                self.assertLessEqual(len(outcome.raw), 200)

    def test_bom_and_zero_width_are_stripped(self) -> None:  # R3-02
        outcome = parse("﻿​ACTION: add(a=2, b=3)")
        self.assertEqual(outcome.args, {"a": "2", "b": "3"})

    def test_fences_with_language_tag_and_inline_backticks(self) -> None:  # R3-03
        for text in (
            "```python\nACTION: add(a=2, b=3)\n```",
            "  ```json  \nACTION: add(a=2, b=3)\n```",
            "`ACTION: add(a=2, b=3)`",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse(text).args, {"a": "2", "b": "3"})

    def test_first_prefix_wins(self) -> None:  # R3-12
        self.assertIsInstance(parse("ACTION: ping()\nFINAL: done"), ParsedAction)
        self.assertIsInstance(parse("FINAL: done\nACTION: ping()"), ParsedFinal)

    def test_final_stops_at_newline(self) -> None:  # R3-10
        self.assertEqual(parse("FINAL:  Paris  \nmore text").text, "Paris")

    def test_word_containing_prefix_is_not_a_prefix(self) -> None:
        self.assertEqual(parse("TRANSACTION: add(a=1)").code, "E_NO_PREFIX")

    def test_escapes(self) -> None:  # R3-24
        outcome = parse(r"ACTION: f(a='it\'s', b=" '"x\\ny\\tz\\\\", c="\\d")')
        self.assertEqual(outcome.args, {"a": "it's", "b": "x\ny\tz\\", "c": "\\d"})

    def test_ambiguous_values_fail(self) -> None:  # S3-3
        for text in (
            'ACTION: f(a="x"y)',
            'ACTION: f(a=x"y")',
            "ACTION: f(a=1,,b=2)",
            "ACTION: f(,)",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse(text).code, "E_BAD_SYNTAX")

    def test_values_are_strings(self) -> None:  # R3-30
        outcome = parse("ACTION: f(a=1, b=true, c=2.5)")
        self.assertTrue(all(isinstance(value, str) for value in outcome.args.values()))

    def test_raw_is_capped(self) -> None:  # R3-31
        self.assertEqual(len(parse("FINAL: " + "x" * 2000).raw), 500)

    def test_input_is_truncated(self) -> None:  # R3-32
        outcome = parse("x" * 8000 + "ACTION: ping()")
        self.assertEqual(outcome.code, "E_NO_PREFIX")


def _random_text(rng: random.Random) -> str:
    kind = rng.randrange(5)
    if kind == 0:
        return "".join(rng.choice(string.printable) for _ in range(rng.randrange(80)))
    if kind == 1:
        return "".join(chr(rng.randrange(0, 0x250)) for _ in range(rng.randrange(80)))
    if kind == 2:
        body = "".join(rng.choice('ab=,()"\'\\ 12') for _ in range(rng.randrange(60)))
        return f"ACTION: {rng.choice(['add', 'search', 'get_weather', 'set_alarm', 'x'])}({body}"
    if kind == 3:
        return rng.choice(["ACTION:", "FINAL:", "```", "**ACTION:**"]) + "".join(
            rng.choice(string.printable) for _ in range(rng.randrange(40))
        )
    return "".join(rng.choice(string.printable) for _ in range(8192))


class ParserRobustnessTest(unittest.TestCase):
    def test_never_raises_fuzz(self) -> None:  # A3-5
        rng = random.Random(20261012)
        for _ in range(10_000):
            text = _random_text(rng)
            outcome = parse(text)
            self.assertIsInstance(outcome, (ParsedAction, ParsedFinal, ParseFailure))
            if isinstance(outcome, ParsedAction):
                result = execute(outcome, fixture_registry, timeout_s=1.0)
                self.assertIn(result.ok, (True, False))

    def test_redos_guard(self) -> None:  # A3-6
        adversarial = (
            'ACTION: f(' + '",' * 4000,
            "a" * 8000,
            "ACTION: f(" + "(" * 8000,
            "ACTION: f(" + "a=" * 4000,
            "ACTION:" * 1100,
            "*" * 8000 + "ACTION",
            "```" * 2666,
            "ACTION: f('" + "\\" * 7990,
        )
        for text in adversarial:
            with self.subTest(text=text[:20]):
                started = time.perf_counter()
                parse(text)
                self.assertLess((time.perf_counter() - started) * 1000, 50)


if __name__ == "__main__":
    unittest.main()
