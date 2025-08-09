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

def evaluate_condition_node(variables: Dict[str, Any], node: Any) -> bool:
    """
    Evaluate a condition node which can be either a leaf comparison or a
    composed boolean (all/any/not). Uses duck-typing to avoid import cycles.
    """
    # Leaf: has var/op/value
    if hasattr(node, "var"):
        return evaluate_condition(
            variables=variables,
            var=getattr(node, "var"),
            op=getattr(node, "op", "truthy"),
            value=getattr(node, "value", None),
        )

    # all: list of child nodes
    if hasattr(node, "all"):
        children = getattr(node, "all") or []
        return all(evaluate_condition_node(variables, child) for child in children)

    # any: list of child nodes
    if hasattr(node, "any"):
        children = getattr(node, "any") or []
        return any(evaluate_condition_node(variables, child) for child in children)

    # not_: single child node
    if hasattr(node, "not_"):
        child = getattr(node, "not_")
        return not evaluate_condition_node(variables, child)

    # Unknown node
    return False

