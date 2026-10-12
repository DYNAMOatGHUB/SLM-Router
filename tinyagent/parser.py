"""Regex parsing layer: model text -> ParsedAction | ParsedFinal | ParseFailure.

Implements PRD-03 Section 3. ``parse`` is pure and deterministic, never raises
for any input, and never converts argument types (values are always ``str``).
Anything ambiguous is a ``ParseFailure``; the parser does not guess.
"""

from __future__ import annotations

import re

from tinyagent.types import ParsedAction, ParsedFinal, ParseFailure, ParseOutcome

MAX_INPUT_CHARS = 8000
MAX_RAW_CHARS = 500

_INVISIBLE = re.compile("[​﻿]")
_FENCE_LINE = re.compile(r"^[ \t]*```[A-Za-z0-9_+.-]*[ \t]*\r?$\n?", re.MULTILINE)
# R3-05, extended to tolerate markdown emphasis directly around the prefix
# (`**ACTION:**`, `__FINAL__:`). The lookbehind plays the role of `\b` but also
# treats `_` as a boundary so that `__ACTION:` is recognised.
_PREFIX = re.compile(r"(?i)(?<![A-Za-z0-9])(ACTION|FINAL)[*_]*[ \t]*:[*_]*")
_ANOTHER_ACTION = re.compile(r"(?i)(?<![A-Za-z0-9])ACTION[*_]*[ \t]*:")
_CALL = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_KEYWORD = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)", re.DOTALL)
_ESCAPES = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t"}
_QUOTES = ('"', "'")


def _normalize(text: str) -> str:
    """R3-02 and R3-03: drop invisible characters and markdown fences."""
    text = _INVISIBLE.sub("", text)
    text = _FENCE_LINE.sub("", text)
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == "`" and stripped[-1] == "`":
        stripped = stripped.strip("`").strip()
    return stripped


def _split_args(text: str, start: int) -> tuple[list[str], int] | None:
    """R3-21: quote-aware scan from just after `(`.

    Returns the comma-separated segments and the index of the closing `)`,
    or None if the input ends before a `)` outside quotes (R3-22).
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in _QUOTES:
            quote = char
            current.append(char)
        elif char == ",":
            segments.append("".join(current))
            current = []
        elif char == ")":
            segments.append("".join(current))
            return segments, index
        else:
            current.append(char)
    return None


def _unquote(value: str) -> str | None:
    """R3-24: decode one complete quoted string, or None if it is not exactly one."""
    quote = value[0]
    chars: list[str] = []
    index = 1
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            following = value[index + 1]
            if following in _ESCAPES:
                chars.append(_ESCAPES[following])
            else:
                chars.append(char + following)
            index += 2
            continue
        if char == quote:
            # The closing quote must be the last character of the value.
            return "".join(chars) if index == len(value) - 1 else None
        chars.append(char)
        index += 1
    return None


def _decode_value(raw: str) -> str | None:
    """R3-24 / R3-25: quoted or bare value; None means bad syntax."""
    value = raw.strip()
    if not value:
        return None
    if value[0] in _QUOTES:
        return _unquote(value)
    if any(quote in value for quote in _QUOTES):
        return None  # Mixed bare and quoted text is ambiguous.
    return value


def _parse_args(segments: list[str]) -> dict[str, str] | None:
    """R3-23 and R3-26 to R3-28: bind segments to positional or keyword names."""
    if len(segments) == 1 and not segments[0].strip():
        return {}  # f() or f( )
    if not segments[-1].strip():
        segments = segments[:-1]  # R3-28 trailing comma; f(,) still fails below
    args: dict[str, str] = {}
    positional = 0
    seen_keyword = False
    for segment in segments:
        keyword = _KEYWORD.match(segment)
        if keyword is not None:
            name, raw_value = keyword.group(1), keyword.group(2)
            if name in args:
                return None  # R3-27 duplicate keyword
            seen_keyword = True
        else:
            if seen_keyword:
                return None  # R3-26 positional after keyword
            name, raw_value = f"#{positional}", segment
            positional += 1
        value = _decode_value(raw_value)
        if value is None:
            return None
        args[name] = value
    return args


def _parse_action(body: str, raw: str) -> ParseOutcome:
    call = _CALL.match(body)
    if call is None:
        return ParseFailure("E_BAD_SYNTAX", raw)  # R3-20
    scanned = _split_args(body, call.end())
    if scanned is None:
        return ParseFailure("E_BAD_SYNTAX", raw)  # R3-22
    segments, close = scanned
    args = _parse_args(segments)
    if args is None:
        return ParseFailure("E_BAD_SYNTAX", raw)
    trailing = body[close + 1:]
    warnings = ("multiple_actions",) if _ANOTHER_ACTION.search(trailing) else ()  # R3-29
    return ParsedAction(call.group(1), args, raw, warnings)


def parse(text: str) -> ParseOutcome:
    """Parse one model reply. Never raises; see PRD-03 Section 3."""
    if not isinstance(text, str):
        return ParseFailure("E_EMPTY", repr(text)[:200])  # R3-01
    raw = text[:MAX_RAW_CHARS]  # R3-31
    normalized = _normalize(text[:MAX_INPUT_CHARS])  # R3-32
    if not normalized:
        return ParseFailure("E_EMPTY", raw)  # R3-04
    prefix = _PREFIX.search(normalized)
    if prefix is None:
        return ParseFailure("E_NO_PREFIX", raw)  # R3-06
    body = normalized[prefix.end():]
    if prefix.group(1).upper() == "ACTION":
        return _parse_action(body, raw)
    final = body.split("\n", 1)[0].strip()  # R3-10
    if not final:
        return ParseFailure("E_BAD_SYNTAX", raw)  # R3-11
    return ParsedFinal(final, raw)
