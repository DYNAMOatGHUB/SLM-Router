"""Strict string-to-value coercion for tool arguments (PRD-03 Section 4).

Coercion never guesses: an input that does not match a rule exactly raises
``CoercionError``. Only ``int()``, ``float()``, ``math.isfinite`` and fixed
lookup tables are used on argument text (R3-42).
"""

from __future__ import annotations

import math
import re
from typing import Any

from slmrouter.errors import CoercionError
from slmrouter.types import ParamSpec

# ASCII digits only: int() would also accept other Unicode digit scripts.
_INT = re.compile(r"([+-]?[0-9]+)(?:\.0+)?")
_MAX_INT_DIGITS = 30
_TRUE = frozenset({"true", "yes", "1", "on"})
_FALSE = frozenset({"false", "no", "0", "off"})
_NONE = frozenset({"none", "null"})


def _coerce_int(text: str) -> int:
    match = _INT.fullmatch(text)
    if match is None:
        raise CoercionError("not an integer")
    digits = match.group(1)
    if len(digits.lstrip("+-")) > _MAX_INT_DIGITS:
        raise CoercionError(f"integer longer than {_MAX_INT_DIGITS} digits")
    return int(digits)


def _coerce_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise CoercionError("not a number") from None
    if not math.isfinite(value):
        raise CoercionError("number is not finite")
    return value


def _coerce_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise CoercionError("not a boolean")


_CONVERTERS = {"int": _coerce_int, "float": _coerce_float, "bool": _coerce_bool}


def coerce(raw: str, param: ParamSpec) -> Any:
    """Convert raw argument text to ``param.type`` or raise ``CoercionError``."""
    if not isinstance(raw, str):
        raise CoercionError("argument is not a string")
    text = raw.strip()
    if param.optional and text.lower() in _NONE:
        return None
    if param.type == "str":
        return raw
    converter = _CONVERTERS.get(param.type)
    if converter is None:
        raise CoercionError(f"unsupported parameter type {param.type!r}")
    return converter(text)
