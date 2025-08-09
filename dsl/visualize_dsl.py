from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Sequence as Seq, Tuple

import dacite
import yaml

from dsl.workflow import (
    ActivityInvocation,
    ActivityStatement,
    IfStatement,
    ParallelStatement,
    SequenceStatement,
    Statement,
    DSLInput,
)
from dsl.yaml_helpers import normalize_if_else_keys


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, Dict[str, Any]]] = []
        self._counter: int = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_node(self, label: str, shape: str = "box", **attrs: Any) -> str:
        node_id = self._next_id("n")
        node_attrs = {"label": label, "shape": shape}
        node_attrs.update(attrs)
        self.nodes[node_id] = node_attrs
        return node_id

    def add_edge(self, src: str, dst: str, label: Optional[str] = None) -> None:
        self.edges.append((src, dst, {"label": label} if label else {}))


def condition_to_text(node: Any) -> str:
    # Leaf condition: has var/op/value
    if hasattr(node, "var"):
        var = getattr(node, "var")
        op = getattr(node, "op", "truthy")
        value = getattr(node, "value", None)
        if value is None or op == "truthy":
            return f"{var} {op}"
        # Render sequences compactly
        if isinstance(value, (list, tuple, set)):
            value_str = ", ".join(map(str, value))
            return f"{var} {op} [{value_str}]"
        return f"{var} {op} {value}"

    # all/any/not
    if hasattr(node, "all"):
        parts = [condition_to_text(child) for child in getattr(node, "all") or []]
        return " AND ".join(parts) if parts else "ALL []"
    if hasattr(node, "any"):
        parts = [condition_to_text(child) for child in getattr(node, "any") or []]
        return " OR ".join(parts) if parts else "ANY []"
    if hasattr(node, "not_"):
        return f"NOT ({condition_to_text(getattr(node, 'not_'))})"
    # Fallback for unknown nodes
    return "<cond>"


def format_activity_label(inv: ActivityInvocation) -> str:
    args = ", ".join(inv.arguments or [])
    res = f" -> {inv.result}" if inv.result else ""
    return f"{inv.name}({args}){res}"


def build_subgraph(graph: Graph, stmt: Statement) -> Tuple[str, List[str]]:
    """
    Build a subgraph for the statement and return (entry_node, exit_nodes).

    exit_nodes is a list so callers can connect merges/joins appropriately.
    """
    if isinstance(stmt, ActivityStatement):
        nid = graph.add_node(
            format_activity_label(stmt.activity),
            shape="box",
            style="filled",
            fillcolor="#e3f2fd",
        )
        return nid, [nid]

    if isinstance(stmt, SequenceStatement):
        entry: Optional[str] = None
        exits: List[str] = []
        for i, elem in enumerate(stmt.sequence.elements):
            elem_entry, elem_exits = build_subgraph(graph, elem)
            if entry is None:
                entry = elem_entry
            # Chain previous exits to this element's entry
            for prev in exits:
                graph.add_edge(prev, elem_entry)
            exits = elem_exits
        # Empty sequence guard
        if entry is None:
            dummy = graph.add_node("(empty)", shape="circle")
            return dummy, [dummy]
        return entry, exits

    if isinstance(stmt, ParallelStatement):
        split = graph.add_node(
            "PARALLEL SPLIT", shape="circle", style="filled", fillcolor="#d1fae5"
        )
        join = graph.add_node(
            "PARALLEL JOIN", shape="circle", style="filled", fillcolor="#d1fae5"
        )
        for branch in stmt.parallel.branches:
            b_entry, b_exits = build_subgraph(graph, branch)
            graph.add_edge(split, b_entry)
            for b_exit in b_exits:
                graph.add_edge(b_exit, join)
        return split, [join]

    if isinstance(stmt, IfStatement):
        decision = graph.add_node(
            "IF", shape="diamond", style="filled", fillcolor="#fff4ce"
        )
        merge = graph.add_node(
            "MERGE", shape="circle", style="filled", fillcolor="#e2e8f0"
        )
        for clause in stmt.if_ladder.clauses:
            cond_text = condition_to_text(clause.condition)
            t_entry, t_exits = build_subgraph(graph, clause.then)
            graph.add_edge(decision, t_entry, label=cond_text)
            for t_exit in t_exits:
                graph.add_edge(t_exit, merge)
        # else
        if getattr(stmt.if_ladder, "else_", None) is not None:
            e_entry, e_exits = build_subgraph(graph, stmt.if_ladder.else_)
            graph.add_edge(decision, e_entry, label="else")
            for e_exit in e_exits:
                graph.add_edge(e_exit, merge)
        else:
            # No else: fall-through from decision to merge
            graph.add_edge(decision, merge, label="else")
        return decision, [merge]

    # Unknown node type - render a placeholder
    label = type(stmt).__name__
    nid = graph.add_node(label, shape="box")
    return nid, [nid]


def to_dot(graph: Graph, title: str) -> str:
    def esc(s: str) -> str:
        return s.replace("\"", "\\\"")

    lines: List[str] = ["digraph G {"]
    # Global graph look-and-feel
    lines.append("  rankdir=TB;")
    lines.append("  bgcolor=\"white\";")
    lines.append("  ranksep=0.7; nodesep=0.5;")
    lines.append("  labelloc=t; labeljust=c;")
    lines.append(f"  label=\"{esc(title)}\";")
    lines.append("  node [fontname=Helvetica, fontsize=10, color=\"#2b6cb0\", style=filled, fillcolor=\"#e3f2fd\"];")
    lines.append("  edge [fontname=Helvetica, fontsize=10, color=\"#4a5568\", arrowsize=0.8, penwidth=1.2];")
    for nid, attrs in graph.nodes.items():
        shape = attrs.get("shape", "box")
        label = esc(str(attrs.get("label", nid)))
        style = attrs.get("style")
        fillcolor = attrs.get("fillcolor")
        style_attr = f", style={style}" if style else ""
        fill_attr = f", fillcolor=\"{fillcolor}\"" if fillcolor else ""
        lines.append(f"  {nid} [shape={shape}, label=\"{label}\"{style_attr}{fill_attr}];\n")
    for src, dst, attrs in graph.edges:
        if attrs.get("label"):
            # Center the label on the edge path (no offset), so it sits on top of the arrow
            lines.append(
                f"  {src} -> {dst} [label=\"{esc(str(attrs['label']))}\", labelfontcolor=\"#2d3748\"];\n"
            )
        else:
            lines.append(f"  {src} -> {dst};\n")
    lines.append("}")
    return "\n".join(lines)


def render_with_graphviz(dot_path: str, png_path: str) -> bool:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return False
    cmd = [dot_bin, "-Tpng", dot_path, "-o", png_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def render_with_matplotlib(graph: Graph, png_path: str, title: str) -> bool:
    try:
        import math
        import matplotlib.pyplot as plt
    except Exception:
        return False

    # Build adjacency and compute naive longest-path levels (topological-like)
    successors: Dict[str, List[str]] = {}
    predecessors_count: Dict[str, int] = {nid: 0 for nid in graph.nodes}
    for src, dst, _ in graph.edges:
        successors.setdefault(src, []).append(dst)
        predecessors_count[dst] = predecessors_count.get(dst, 0) + 1
        predecessors_count.setdefault(src, predecessors_count.get(src, 0))

    # Find entry candidates (no predecessors). If none, still seed with an arbitrary node
    levels: Dict[str, int] = {nid: 0 for nid, cnt in predecessors_count.items() if cnt == 0}
    if not levels and graph.nodes:
        # Graph with cycles or fully connected; seed first node
        first_nid = next(iter(graph.nodes.keys()))
        levels[first_nid] = 0

    # Relaxation to assign levels
    changed = True
    while changed:
        changed = False
        for src, dst, _ in graph.edges:
            src_lvl = levels.get(src, 0)
            dst_lvl = levels.get(dst, 0)
            if dst_lvl < src_lvl + 1:
                levels[dst] = src_lvl + 1
                changed = True
            levels.setdefault(src, src_lvl)

    # Bucket nodes by level
    level_to_nodes: Dict[int, List[str]] = {}
    for nid, lvl in levels.items():
        level_to_nodes.setdefault(lvl, []).append(nid)
    max_level = max(level_to_nodes.keys()) if level_to_nodes else 0

    # Simple grid positions
    positions: Dict[str, Tuple[float, float]] = {}
    for lvl in range(max_level + 1):
        row = level_to_nodes.get(lvl, [])
        n = max(1, len(row))
        for i, nid in enumerate(row):
            x = (i + 1) / (n + 1)
            y = -lvl
            positions[nid] = (x, y)

    # Figure size scaled by depth and max breadth to reduce overlaps
    max_breadth = max((len(v) for v in level_to_nodes.values()), default=1)
    fig_width = max(6.0, 1.5 * max_breadth)
    fig_height = max(4.0, 1.2 * (max_level + 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_title(title)
    ax.axis("off")

    # Draw edges first
    for src, dst, attrs in graph.edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"),
        )
        if attrs.get("label"):
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ax.text(mx, my + 0.05, str(attrs["label"]), ha="center", va="bottom", fontsize=8)

    # Determine axes limits so all nodes are visible
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin_x, margin_y = 0.2, 0.5
        ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
        ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    # Draw nodes as boxes
    for nid, attrs in graph.nodes.items():
        x, y = positions.get(nid, (0.5, 0))
        label = str(attrs.get("label", nid))
        # box
        width, height = 0.2, 0.1 + 0.02 * max(1, len(label) // 12)
        rect = plt.Rectangle((x - width / 2, y - height / 2), width, height, fc="#eaf2fb", ec="#2b6cb0")
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, wrap=True)

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return True


def visualize_yaml(yaml_path: str, out_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Create DOT and PNG for a given YAML workflow file.
    Returns (dot_path, png_path_or_none_if_failed)
    """
    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    normalized = normalize_if_else_keys(loaded)
    dsl_input = dacite.from_dict(DSLInput, normalized)

    graph = Graph()
    entry, _ = build_subgraph(graph, dsl_input.root)
    # Ensure entry is visually marked
    graph.nodes[entry]["label"] = "START → " + str(graph.nodes[entry]["label"])

    title = os.path.basename(yaml_path)
    dot_text = to_dot(graph, title)

    out_dir = out_dir or os.path.dirname(yaml_path)
    base = os.path.splitext(os.path.basename(yaml_path))[0]
    dot_path = os.path.join(out_dir, f"{base}.dot")
    png_path = os.path.join(out_dir, f"{base}.png")

    with open(dot_path, "w") as f:
        f.write(dot_text)

    # Prefer Graphviz if available
    if render_with_graphviz(dot_path, png_path):
        return dot_path, png_path

    # Fallback to very simple matplotlib renderer
    if render_with_matplotlib(graph, png_path, title):
        return dot_path, png_path

    # No renderer available
    return dot_path, None


def main(argv: Optional[Seq[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize DSL workflow YAML as a DAG (DOT/PNG)")
    parser.add_argument("yamls", nargs="*", help="Paths to YAML workflow files")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for images and DOT")
    args = parser.parse_args(argv)

    yaml_paths: List[str]
    if args.yamls:
        yaml_paths = list(args.yamls)
    else:
        # Default to workflow1-4 in the same directory
        here = os.path.dirname(__file__)
        yaml_paths = [
            os.path.join(here, f"workflow{i}.yaml") for i in range(1, 5)
            if os.path.exists(os.path.join(here, f"workflow{i}.yaml"))
        ]
        if not yaml_paths:
            print("No YAML files provided and none of workflow1-4.yaml found.")
            return 2

    os.makedirs(args.out_dir or os.path.dirname(yaml_paths[0]) or ".", exist_ok=True)

    rc = 0
    for path in yaml_paths:
        try:
            dot_path, png_path = visualize_yaml(path, out_dir=args.out_dir)
            if png_path:
                print(f"Rendered: {path} -> {png_path} (and {dot_path})")
            else:
                print(
                    "Generated DOT but could not render PNG (install Graphviz 'dot' or matplotlib):\n"
                    f"  DOT: {dot_path}"
                )
        except Exception as exc:  # pragma: no cover - visualization helper shouldn't break tests
            rc = 1
            print(f"Failed to visualize {path}: {exc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


