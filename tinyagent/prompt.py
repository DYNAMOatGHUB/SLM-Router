import math
import re
import unicodedata
from typing import Sequence
from .types import ToolSpec

PROMPT_TEMPLATE = """Pick one tool for the query.
Tools:
{tool_lines}
Reply with ONE line, either:
ACTION: tool_name(arg="value")
FINAL: short answer (only if no tool fits)
Query: {query}
Reply:"""

STOP_SEQUENCES = ("\n\n", "\nQuery:", "\nReply:")

_REPAIR_REASONS = {
    "E_EMPTY": "empty reply",
    "E_NO_PREFIX": "reply must start with ACTION: or FINAL:",
    "E_BAD_SYNTAX": "malformed call",
    "E_UNKNOWN_TOOL": 'unknown tool "{detail}"',
    "E_MISSING_ARG": 'missing arg "{detail}"',
    "E_UNEXPECTED_ARG": 'unknown arg "{detail}"',
    "E_TYPE_COERCION": 'wrong type for "{detail}"',
    "E_FINAL_NOT_ALLOWED": "a tool call is required",
}

def estimate_tokens(text: str) -> int:
    """Estimate token count for budgeting (not a real tokenizer)."""
    return math.ceil(len(text) / 4)

def _sanitize(text: str) -> str:
    chars = []
    for c in text:
        if unicodedata.category(c) == "Cc":
            chars.append(" ")
        else:
            chars.append(c)
    s = "".join(chars)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def render_tool_line(spec: ToolSpec) -> str:
    rendered_params = []
    for p in spec.params:
        if p.required:
            rendered_params.append(f"{p.name}:{p.type}")
        else:
            if p.default is None:
                default_str = "None"
            elif isinstance(p.default, str):
                default_val = _sanitize(p.default)
                if len(default_val) > 20:
                    default_val = default_val[:20] + "..."
                default_str = f'"{default_val}"'
            else:
                default_str = repr(p.default)
            rendered_params.append(f"{p.name}:{p.type}={default_str}")
            
    params_str = ", ".join(rendered_params)
    return f"{spec.name}({params_str}) - {spec.description}"

def render_prompt(query: str, tools: Sequence[ToolSpec], *, max_query_chars: int = 2000) -> str:
    sanitized = _sanitize(query)[:max_query_chars]
    if not sanitized:
        raise ValueError("empty query")
        
    if not tools:
        tool_lines = "(none)"
    else:
        tool_lines = "\n".join(render_tool_line(t) for t in tools)
        
    return PROMPT_TEMPLATE.format(tool_lines=tool_lines, query=sanitized)

def render_repair_prompt(prompt: str, previous_output: str, code: str, detail: str) -> str:
    """
    Renders a repair prompt for a model to try again.
    Note: Repair prompts are not cumulative. Pass the original prompt here.
    """
    sanitized_output = _sanitize(previous_output)[:200]
    sanitized_detail = _sanitize(detail)[:40]
    
    reason_template = _REPAIR_REASONS.get(code, "bad format")
    short_reason = reason_template.replace("{detail}", sanitized_detail)
    
    return f'{prompt} {sanitized_output}\nInvalid: {short_reason}. Reply with ONE line: ACTION: tool_name(arg="value") or FINAL: text\nReply:'
