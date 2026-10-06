import inspect
import re
import types
import typing
from typing import Any, Callable, Optional, Union, get_args, get_origin

from .types import ParamSpec, ToolSpec
from .registry import ToolRegistry, default_registry

_ALLOWED_TYPES = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool"
}

_NAME_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

def _parse_type_annotation(param_name: str, fn_name: str, annotation: Any) -> tuple[str, bool]:
    if annotation is inspect.Parameter.empty:
        return "str", False
        
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    # Handle Union[T, None] and T | None
    is_union = origin is Union or isinstance(annotation, types.UnionType)
    if is_union:
        none_type = type(None)
        valid_args = [a for a in args if a is not none_type]
        if len(valid_args) == 1 and type(None) in args:
            inner_type = valid_args[0]
            if inner_type in _ALLOWED_TYPES:
                return _ALLOWED_TYPES[inner_type], True
        raise TypeError(f"tiny_tool: parameter '{param_name}' of '{fn_name}' has unsupported type {annotation}; allowed: str,int,float,bool")
    
    if annotation in _ALLOWED_TYPES:
        return _ALLOWED_TYPES[annotation], False
        
    raise TypeError(f"tiny_tool: parameter '{param_name}' of '{fn_name}' has unsupported type {annotation}; allowed: str,int,float,bool")

def tiny_tool(func: Callable | None = None, *, name: str | None = None, registry: ToolRegistry | None = None):
    if func is None:
        def wrapper(f: Callable) -> Callable:
            return tiny_tool(f, name=name, registry=registry)
        return wrapper
        
    fn_name = func.__name__
    tool_name = name if name is not None else fn_name
    
    if not _NAME_REGEX.match(tool_name):
        raise ValueError(f"Invalid tool name: '{tool_name}'")
        
    sig = inspect.signature(func)
    type_hints = typing.get_type_hints(func)
    
    params = []
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
            raise TypeError(f"tiny_tool: parameter '{param_name}' of '{fn_name}' has unsupported kind {param.kind}")
            
        annotation = type_hints.get(param_name, inspect.Parameter.empty)
        param_type, is_optional = _parse_type_annotation(param_name, fn_name, annotation)
        
        required = param.default is inspect.Parameter.empty
        default_val = None if required else param.default
        
        params.append(ParamSpec(
            name=param_name,
            type=param_type,
            required=required,
            default=default_val,
            optional=is_optional
        ))
        
    doc = inspect.getdoc(func)
    if doc:
        lines = [line.strip() for line in doc.splitlines() if line.strip()]
        if lines:
            desc = " ".join(lines[0].split())
            if len(desc) > 80:
                desc = desc[:77] + "..."
        else:
            desc = tool_name
    else:
        desc = tool_name
        
    spec = ToolSpec(
        name=tool_name,
        description=desc,
        params=tuple(params),
        func=func
    )
    
    reg = registry if registry is not None else default_registry
    reg.register(spec)
    
    return func
