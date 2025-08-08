from __future__ import annotations

from typing import Any, Dict


def evaluate_condition(
    variables: Dict[str, Any],
    var: str,
    op: str = "truthy",
    value: Any | None = None,
) -> bool:
    """
    Evaluate a simple deterministic condition against workflow variables.

    Supported ops: truthy, eq, ne, lt, gt, le, ge, in, contains
    """
    lhs = variables.get(var)

    if op == "truthy":
        return bool(lhs)
    if op == "eq":
        return lhs == value
    if op == "ne":
        return lhs != value
    if op == "lt":
        return lhs < value
    if op == "gt":
        return lhs > value
    if op == "le":
        return lhs <= value
    if op == "ge":
        return lhs >= value
    if op == "in":
        try:
            return lhs in value  # type: ignore[operator]
        except TypeError:
            return False
    if op == "contains":
        if isinstance(lhs, (list, tuple, set, str, bytes)):
            return value in lhs  # type: ignore[operator]
        return False

    # Unknown operator
    return False


