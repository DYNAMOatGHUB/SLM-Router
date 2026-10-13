"""PRD-03 coercion table (A3-2)."""

from __future__ import annotations

import unittest

from slmrouter.coercion import coerce
from slmrouter.errors import CoercionError
from slmrouter.types import ParamSpec


def _param(type_name: str, optional: bool = False) -> ParamSpec:
    return ParamSpec("x", type_name, True, None, optional)


ACCEPTED = (
    ("str", "hello", "hello"),
    ("str", "", ""),
    ("str", "  padded  ", "  padded  "),
    ("str", "None", "None"),
    ("int", "0", 0),
    ("int", "42", 42),
    ("int", "-7", -7),
    ("int", "+7", 7),
    ("int", " 12 ", 12),
    ("int", "3.0", 3),
    ("int", "-3.000", -3),
    ("int", "007", 7),
    ("int", "9" * 30, int("9" * 30)),
    ("float", "2.5", 2.5),
    ("float", "-0.5", -0.5),
    ("float", "3", 3.0),
    ("float", "1e3", 1000.0),
    ("float", " 4.25 ", 4.25),
    ("bool", "true", True),
    ("bool", "TRUE", True),
    ("bool", "Yes", True),
    ("bool", "1", True),
    ("bool", "on", True),
    ("bool", "false", False),
    ("bool", "No", False),
    ("bool", "0", False),
    ("bool", "OFF", False),
)

REJECTED = (
    ("int", "2.5"),
    ("int", "1e3"),
    ("int", "0x10"),
    ("int", "two"),
    ("int", ""),
    ("int", "1" * 31),
    ("int", "1 000"),
    ("int", "٣"),  # Arabic-Indic digit three
    ("int", "None"),
    ("float", "nan"),
    ("float", "inf"),
    ("float", "-inf"),
    ("float", ""),
    ("float", "1e400"),
    ("float", "two"),
    ("bool", "maybe"),
    ("bool", ""),
    ("bool", "2"),
    ("bool", "y"),
)


class CoercionTest(unittest.TestCase):
    def test_accepted(self) -> None:
        for type_name, raw, expected in ACCEPTED:
            with self.subTest(type=type_name, raw=raw):
                value = coerce(raw, _param(type_name))
                self.assertEqual(value, expected)
                self.assertIs(type(value), type(expected))

    def test_rejected(self) -> None:  # R3-40: no best guess
        for type_name, raw in REJECTED:
            with self.subTest(type=type_name, raw=raw):
                with self.assertRaises(CoercionError) as raised:
                    coerce(raw, _param(type_name))
                self.assertNotIn("\n", str(raised.exception))

    def test_optional_none(self) -> None:
        for type_name in ("str", "int", "float", "bool"):
            for raw in ("none", "None", "NULL", " null "):
                with self.subTest(type=type_name, raw=raw):
                    self.assertIsNone(coerce(raw, _param(type_name, optional=True)))

    def test_optional_still_coerces_values(self) -> None:
        self.assertEqual(coerce("5", _param("int", optional=True)), 5)


if __name__ == "__main__":
    unittest.main()
