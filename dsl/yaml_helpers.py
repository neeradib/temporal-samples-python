from __future__ import annotations

from typing import Any


def normalize_if_else_keys(obj: Any) -> Any:
    """
    Recursively map YAML keys "if" -> "if_" and "else" -> "else_" so they
    bind correctly to Python dataclass fields.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            nk = k
            if k == "if":
                nk = "if_"
            elif k == "else":
                nk = "else_"
            new_obj[nk] = normalize_if_else_keys(v)
        return new_obj
    if isinstance(obj, list):
        return [normalize_if_else_keys(i) for i in obj]
    return obj


