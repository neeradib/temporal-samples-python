from __future__ import annotations

from typing import Any


def normalize_if_else_keys(obj: Any) -> Any:
    """
    Recursively map YAML keys for reserved words so they bind correctly to
    Python dataclass fields.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            nk = k
            if k == "else":
                nk = "else_"
            elif k == "not":
                nk = "not_"
            new_obj[nk] = normalize_if_else_keys(v)
        return new_obj
    if isinstance(obj, list):
        return [normalize_if_else_keys(i) for i in obj]
    return obj


