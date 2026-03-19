"""Pure-SVG computation graph visualization utilities."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from .autodiff import Value


def trace_graph(root: Value) -> tuple[list[Value], list[tuple[Value, Value]]]:
    """Return nodes and directed edges from inputs to root."""

    nodes: list[Value] = []
    edges: list[tuple[Value, Value]] = []
    visited: set[Value] = set()

    def build(node: Value) -> None:
        if node in visited:
            return
        visited.add(node)
        for child in sorted(node._prev, key=id):
            build(child)
            edges.append((child, node))
        nodes.append(node)

    build(root)
    return nodes, edges


def _format_number(value: float) -> str:
    return f"{value:.4f}"


def _node_depths(root: Value) -> dict[Value, int]:
    depths = {root: 0}
    stack = [root]

    while stack:
        node = stack.pop()
        for child in node._prev:
            candidate = depths[node] + 1
            if candidate > depths.get(child, -1):
                depths[child] = candidate
                stack.append(child)

    return depths


def computation_graph_svg(
    root: Value,
    *,
    node_width: int = 200,
    node_height: int = 94,
    horizontal_gap: int = 64,
    vertical_gap: int = 28,
    padding: int = 24,
) -> str:
    """Render a computation graph as an SVG string."""

    nodes, edges = trace_graph(root)
    depths = _node_depths(root)
    order = {node: index for index, node in enumerate(nodes)}
    layers: dict[int, list[Value]] = defaultdict(list)
    for node in nodes:
        layers[depths[node]].append(node)
    for layer in layers.values():
        layer.sort(key=order.__getitem__)

    max_depth = max(depths.values(), default=0)
    max_layer_size = max((len(layer) for layer in layers.values()), default=1)
    width = padding * 2 + (max_depth + 1) * node_width + max_depth * horizontal_gap
    height = padding * 2 + max_layer_size * node_height + (max_layer_size - 1) * vertical_gap

    positions: dict[Value, tuple[float, float]] = {}
    for depth, layer_nodes in layers.items():
        layer_height = len(layer_nodes) * node_height + max(0, len(layer_nodes) - 1) * vertical_gap
        start_y = padding + (height - padding * 2 - layer_height) / 2
        x = padding + (max_depth - depth) * (node_width + horizontal_gap)
        for index, node in enumerate(layer_nodes):
            y = start_y + index * (node_height + vertical_gap)
            positions[node] = (x, y)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Learnify computation graph">',
        f'<rect width="{width}" height="{height}" fill="#fcfcf7" />',
        '<defs>',
        '<marker id="arrow" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">',
        '<path d="M0,0 L10,4 L0,8 z" fill="#4b5563" />',
        "</marker>",
        "</defs>",
    ]

    for parent, child in edges:
        parent_x, parent_y = positions[parent]
        child_x, child_y = positions[child]
        start_x = parent_x + node_width
        start_y = parent_y + node_height / 2
        end_x = child_x
        end_y = child_y + node_height / 2
        svg.append(
            f'<line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}" '
            'stroke="#4b5563" stroke-width="2" marker-end="url(#arrow)" />'
        )

    for node in nodes:
        x, y = positions[node]
        node_label = escape(node.label or f"v{order[node]}")
        op_label = escape(node._op) if node._op else "leaf"
        text_lines = [
            node_label,
            f"data={_format_number(node.data)}",
            f"grad={_format_number(node.grad)}",
            f"op={op_label}",
        ]
        svg.extend(
            [
                f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" '
                'rx="14" fill="#f8fafc" stroke="#0f172a" stroke-width="2" />',
                f'<rect x="{x}" y="{y}" width="{node_width}" height="26" rx="14" '
                'fill="#dbeafe" stroke="none" />',
            ]
        )
        for line_index, line in enumerate(text_lines):
            text_y = y + 18 + line_index * 18
            font_weight = "700" if line_index == 0 else "500"
            font_size = "13" if line_index == 0 else "12"
            svg.append(
                f'<text x="{x + 12}" y="{text_y}" fill="#0f172a" '
                f'font-size="{font_size}" font-family="Menlo, monospace" '
                f'font-weight="{font_weight}">{escape(line)}</text>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def save_computation_graph_svg(root: Value, path: str | Path, **kwargs: int) -> Path:
    """Save an SVG rendering to disk and return the output path."""

    destination = Path(path)
    destination.write_text(computation_graph_svg(root, **kwargs), encoding="utf-8")
    return destination


def computation_graph_dot(root: Value) -> str:
    """Return a Graphviz DOT representation of the computation graph."""

    nodes, edges = trace_graph(root)
    identifiers = {node: f"node_{index}" for index, node in enumerate(nodes)}
    lines = ["digraph ComputationGraph {", "  rankdir=LR;"]
    for node in nodes:
        label = node.label or identifiers[node]
        parts = [
            label.replace('"', '\\"'),
            f"data={_format_number(node.data)}",
            f"grad={_format_number(node.grad)}",
        ]
        if node._op:
            parts.append(f"op={node._op}")
        text = "\\n".join(part.replace('"', '\\"') for part in parts)
        lines.append(f'  {identifiers[node]} [shape=box, style="rounded", label="{text}"];')
    for parent, child in edges:
        lines.append(f"  {identifiers[parent]} -> {identifiers[child]};")
    lines.append("}")
    return "\n".join(lines)
