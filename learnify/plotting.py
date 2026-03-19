"""Model plotting helpers for Learnify."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from .ml.clustering import AgglomerativeClustering, KMeans
from .ml.ensemble import RandomForestClassifier
from .ml.linear import LinearRegressionGD, LogisticRegressionGD
from .ml.svm import LinearSVM
from .ml.tree import DecisionTreeClassifier
from .nn.mlp import MLP


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _get_ax(ax: Axes | None, figsize: tuple[float, float] = (6.0, 4.0)) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    figure, created_ax = plt.subplots(figsize=figsize)
    return figure, created_ax


def _label_names(feature_names: Sequence[str] | None, size: int) -> list[str]:
    if feature_names is None:
        return [f"x{i}" for i in range(size)]
    return list(feature_names)


def plot_loss_curve(
    loss_history: Sequence[float],
    *,
    ax: Axes | None = None,
    title: str = "Training loss",
) -> Figure:
    figure, ax = _get_ax(ax)
    ax.plot(np.arange(1, len(loss_history) + 1), loss_history, color="#0f766e", linewidth=2.5)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    return figure


def plot_linear_coefficients(
    model: LinearRegressionGD | LogisticRegressionGD | LinearSVM,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Weights and bias",
) -> Figure:
    weights = np.asarray(model.weights_, dtype=float).reshape(-1)
    labels = _label_names(feature_names, len(weights)) + ["bias"]
    values = np.concatenate([weights, [float(model.bias_)]])
    colors = ["#2563eb" if value >= 0.0 else "#dc2626" for value in values]

    figure, ax = _get_ax(ax)
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="#0f172a", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Coefficient value")
    ax.grid(axis="y", alpha=0.2)
    return figure


def plot_regression_fit(
    model: LinearRegressionGD,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax: Axes | None = None,
    feature_name: str = "x",
    target_name: str = "y",
    title: str = "Linear regression fit",
) -> Figure:
    X = _ensure_2d(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[1] != 1:
        raise ValueError("plot_regression_fit supports one input feature.")

    figure, ax = _get_ax(ax)
    x_values = X[:, 0]
    dense_x = np.linspace(x_values.min() - 0.3, x_values.max() + 0.3, 200)
    dense_predictions = model.predict(dense_x.reshape(-1, 1))

    ax.scatter(x_values, y, s=70, color="#0f766e", edgecolor="white", linewidth=0.9, label="data")
    ax.plot(dense_x, dense_predictions, color="#1d4ed8", linewidth=2.8, label="fit")
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    equation = f"y = {model.weights_[0]:.2f} * {feature_name} + {model.bias_:.2f}"
    ax.text(
        0.03,
        0.95,
        equation,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
    )
    return figure


def _meshgrid(X: np.ndarray, steps: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[1] != 2:
        raise ValueError("This plot requires exactly two input features.")

    x_margin = max(0.5, (X[:, 0].max() - X[:, 0].min()) * 0.2)
    y_margin = max(0.5, (X[:, 1].max() - X[:, 1].min()) * 0.2)
    x_min, x_max = X[:, 0].min() - x_margin, X[:, 0].max() + x_margin
    y_min, y_max = X[:, 1].min() - y_margin, X[:, 1].max() + y_margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def plot_binary_decision_surface(
    model: LogisticRegressionGD | LinearSVM,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    response: str = "decision",
    title: str = "Decision surface",
) -> Figure:
    X = _ensure_2d(X)
    y = np.asarray(y)
    labels = _label_names(feature_names, X.shape[1])
    figure, ax = _get_ax(ax, figsize=(6.5, 5.0))
    xx, yy, grid = _meshgrid(X)

    if response == "probability":
        surface = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        contour = ax.contourf(xx, yy, surface, levels=18, cmap="RdBu", alpha=0.35)
        ax.contour(xx, yy, surface, levels=[0.5], colors="#0f172a", linewidths=2)
        colorbar_label = "P(class = 1)"
    elif response == "decision":
        surface = model.decision_function(grid).reshape(xx.shape)
        max_abs = float(np.max(np.abs(surface))) or 1.0
        contour = ax.contourf(
            xx,
            yy,
            surface,
            levels=24,
            cmap="coolwarm",
            alpha=0.35,
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        )
        levels = [-1.0, 0.0, 1.0] if isinstance(model, LinearSVM) else [0.0]
        styles = ["--", "-", "--"] if isinstance(model, LinearSVM) else ["-"]
        ax.contour(xx, yy, surface, levels=levels, colors="#0f172a", linewidths=2, linestyles=styles)
        colorbar_label = "w·x + b"
    else:
        raise ValueError("response must be 'decision' or 'probability'")

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=75,
        edgecolor="white",
        linewidth=0.9,
    )
    figure.colorbar(contour, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.15)
    scatter.set_clim(float(np.min(y)), float(np.max(y)))
    return figure


def plot_discriminant_function(
    model: LogisticRegressionGD | LinearSVM,
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Discriminant function",
) -> Figure:
    X = _ensure_2d(X)
    labels = _label_names(feature_names, X.shape[1])
    figure, ax = _get_ax(ax, figsize=(6.5, 5.0))
    xx, yy, grid = _meshgrid(X)
    scores = model.decision_function(grid).reshape(xx.shape)
    max_abs = float(np.max(np.abs(scores))) or 1.0
    contour = ax.contourf(
        xx,
        yy,
        scores,
        levels=24,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    levels = [-1.0, 0.0, 1.0] if isinstance(model, LinearSVM) else [0.0]
    ax.contour(xx, yy, scores, levels=levels, colors="black", linewidths=1.8)
    if y is not None:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=70, edgecolor="white", linewidth=0.9)
    figure.colorbar(contour, ax=ax, label="w·x + b")
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.15)
    return figure


def _class_palette(size: int) -> list[str]:
    palette = ["#60a5fa", "#f59e0b", "#34d399", "#f87171", "#a78bfa", "#2dd4bf"]
    return [palette[index % len(palette)] for index in range(size)]


def _tree_positions(root: object) -> tuple[dict[int, tuple[float, float]], int]:
    positions: dict[int, tuple[float, float]] = {}

    def layout(node: object, depth: int, next_x: float) -> float:
        key = id(node)
        if node.is_leaf:
            positions[key] = (next_x + 0.5, -depth)
            return next_x + 1.0

        current_x = layout(node.left, depth + 1, next_x)
        current_x = layout(node.right, depth + 1, current_x)
        left_x = positions[id(node.left)][0]
        right_x = positions[id(node.right)][0]
        positions[key] = ((left_x + right_x) / 2.0, -depth)
        return current_x

    max_x = layout(root, 0, 0.0)
    return positions, math.ceil(max_x)


def _tree_node_label(
    tree: DecisionTreeClassifier,
    node: object,
    feature_names: Sequence[str] | None,
    class_names: Sequence[str] | None,
) -> str:
    classes = list(tree.classes_ if class_names is None else class_names)
    predicted = classes[node.predicted_index]
    counts = ", ".join(str(int(value)) for value in node.class_counts)
    if node.is_leaf:
        return f"leaf\npredict={predicted}\ncounts=[{counts}]"

    feature_label = _label_names(feature_names, tree.n_features_in_)[node.feature_index]
    return (
        f"{feature_label} <= {node.threshold:.2f}\n"
        f"predict={predicted}\n"
        f"counts=[{counts}]"
    )


def plot_decision_tree(
    tree: DecisionTreeClassifier,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Decision tree",
) -> Figure:
    figure, ax = _get_ax(ax, figsize=(8.0, 5.5))
    root = tree.root_
    positions, width = _tree_positions(root)
    palette = _class_palette(len(tree.classes_))

    def draw(node: object) -> None:
        x, y = positions[id(node)]
        purity = float(np.max(node.class_counts) / np.sum(node.class_counts))
        facecolor = palette[node.predicted_index]
        ax.text(
            x,
            y,
            _tree_node_label(tree, node, feature_names, class_names),
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": facecolor,
                "alpha": 0.18 + 0.45 * purity,
                "edgecolor": "#0f172a",
            },
        )
        if node.is_leaf:
            return
        for child, edge_label in ((node.left, "yes"), (node.right, "no")):
            child_x, child_y = positions[id(child)]
            ax.plot([x, child_x], [y - 0.08, child_y + 0.18], color="#475569", linewidth=1.6)
            ax.text((x + child_x) / 2, (y + child_y) / 2, edge_label, fontsize=8, color="#334155")
            draw(child)

    draw(root)
    ax.set_title(title)
    ax.set_xlim(0, width + 0.5)
    ax.set_ylim(-tree.depth_ - 0.8, 0.8)
    ax.axis("off")
    return figure


def plot_tree_feature_importances(
    tree: DecisionTreeClassifier,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Decision tree feature importance",
) -> Figure:
    figure, ax = _get_ax(ax)
    labels = _label_names(feature_names, tree.n_features_in_)
    ax.bar(labels, tree.feature_importances_, color="#7c3aed", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Normalized impurity decrease")
    ax.set_ylim(0.0, max(0.05, float(np.max(tree.feature_importances_)) * 1.15))
    ax.grid(axis="y", alpha=0.2)
    return figure


def plot_random_forest_feature_importances(
    forest: RandomForestClassifier,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Random forest feature importance",
) -> Figure:
    figure, ax = _get_ax(ax)
    labels = _label_names(feature_names, forest.n_features_in_)
    ax.bar(labels, forest.feature_importances_, color="#ea580c", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Average tree importance")
    ax.grid(axis="y", alpha=0.2)
    return figure


def plot_random_forest(
    forest: RandomForestClassifier,
    *,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    max_trees: int = 4,
    title: str = "Random forest overview",
) -> Figure:
    tree_count = min(max_trees, len(forest.trees_))
    figure = plt.figure(figsize=(5.0 * tree_count, 8.0))
    grid = figure.add_gridspec(2, tree_count, height_ratios=[3.0, 1.3])

    for index in range(tree_count):
        axis = figure.add_subplot(grid[0, index])
        plot_decision_tree(
            forest.trees_[index],
            ax=axis,
            feature_names=feature_names,
            class_names=class_names,
            title=f"Tree {index + 1}",
        )

    importance_ax = figure.add_subplot(grid[1, :])
    plot_random_forest_feature_importances(
        forest,
        ax=importance_ax,
        feature_names=feature_names,
        title="Average feature importance across the forest",
    )
    figure.suptitle(title, y=0.98, fontsize=15)
    figure.tight_layout()
    return figure


def plot_kmeans_clusters(
    model: KMeans,
    X: np.ndarray,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "KMeans clustering",
) -> Figure:
    X = _ensure_2d(X)
    labels = _label_names(feature_names, X.shape[1])
    figure, ax = _get_ax(ax, figsize=(6.0, 5.0))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=80, edgecolor="white")
    ax.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=220,
        color="#ef4444",
        edgecolor="black",
        linewidth=1.0,
        label="centroids",
    )
    figure.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{title} (inertia={model.inertia_:.3f})")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return figure


def plot_agglomerative_clusters(
    model: AgglomerativeClustering,
    X: np.ndarray,
    *,
    ax: Axes | None = None,
    feature_names: Sequence[str] | None = None,
    title: str = "Agglomerative clustering",
) -> Figure:
    X = _ensure_2d(X)
    labels = _label_names(feature_names, X.shape[1])
    figure, ax = _get_ax(ax, figsize=(6.0, 5.0))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", s=80, edgecolor="white")
    figure.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.2)
    return figure


def plot_agglomerative_dendrogram(
    model: AgglomerativeClustering,
    *,
    ax: Axes | None = None,
    leaf_labels: Sequence[str] | None = None,
    title: str = "Agglomerative merge diagram",
) -> Figure:
    if not hasattr(model, "children_") or len(model.children_) == 0:
        raise ValueError("The fitted clustering does not contain merge history to draw.")

    figure, ax = _get_ax(ax, figsize=(7.0, 4.5))
    n_samples = len(model.labels_)
    x_positions = {index: float(index) for index in range(n_samples)}
    y_positions = {index: 0.0 for index in range(n_samples)}

    for merge_index, ((left, right), distance) in enumerate(zip(model.children_, model.distances_, strict=True)):
        left_x, right_x = x_positions[int(left)], x_positions[int(right)]
        left_y, right_y = y_positions[int(left)], y_positions[int(right)]
        ax.plot([left_x, left_x], [left_y, distance], color="#334155", linewidth=1.7)
        ax.plot([right_x, right_x], [right_y, distance], color="#334155", linewidth=1.7)
        ax.plot([left_x, right_x], [distance, distance], color="#334155", linewidth=1.7)
        new_id = n_samples + merge_index
        x_positions[new_id] = (left_x + right_x) / 2.0
        y_positions[new_id] = float(distance)

    ax.set_title(title)
    ax.set_ylabel("Merge distance")
    ax.set_xlabel("Sample")
    ax.set_xticks(range(n_samples))
    if leaf_labels is None:
        ax.set_xticklabels([str(index) for index in range(n_samples)])
    else:
        ax.set_xticklabels(list(leaf_labels), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    return figure


def plot_mlp_architecture(
    model: MLP,
    *,
    ax: Axes | None = None,
    title: str = "MLP architecture",
) -> Figure:
    figure, ax = _get_ax(ax, figsize=(8.0, 5.0))
    sizes = [model.n_inputs_, *model.layer_sizes_]
    max_width = max(sizes)
    layer_x = np.linspace(0.0, 1.0, len(sizes))
    layer_positions: list[list[tuple[float, float]]] = []

    for layer_index, size in enumerate(sizes):
        if size == 1:
            y_values = [0.5]
        else:
            y_values = np.linspace(0.1, 0.9, size)
        positions = [(layer_x[layer_index], y_value) for y_value in y_values]
        layer_positions.append(positions)

    all_weights = [
        abs(weight.data)
        for layer in model.layers
        for neuron in layer.neurons
        for weight in neuron.weights
    ]
    max_abs_weight = max(all_weights, default=1.0)

    for layer_index, layer in enumerate(model.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            end_x, end_y = layer_positions[layer_index + 1][neuron_index]
            for input_index, weight in enumerate(neuron.weights):
                start_x, start_y = layer_positions[layer_index][input_index]
                color = "#2563eb" if weight.data >= 0.0 else "#dc2626"
                linewidth = 0.8 + 3.0 * abs(weight.data) / max_abs_weight
                alpha = 0.25 + 0.6 * abs(weight.data) / max_abs_weight
                ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=linewidth, alpha=alpha)

    for layer_index, positions in enumerate(layer_positions):
        for node_index, (x, y) in enumerate(positions):
            facecolor = "#e2e8f0" if layer_index == 0 else "#dbeafe"
            if layer_index == len(layer_positions) - 1:
                facecolor = "#dcfce7"
            ax.scatter([x], [y], s=950 / max(1.0, max_width / 4), color=facecolor, edgecolor="#0f172a", zorder=3)
            if layer_index == 0:
                label = f"x{node_index}"
            elif layer_index == len(layer_positions) - 1:
                label = f"y{node_index}"
            else:
                label = f"h{layer_index}_{node_index}"
            ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4)

    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.plot([], [], color="#2563eb", linewidth=2, label="positive weight")
    ax.plot([], [], color="#dc2626", linewidth=2, label="negative weight")
    ax.legend(frameon=False, loc="lower center", ncol=2)
    return figure


def plot_mlp_weight_matrices(model: MLP, *, title: str = "MLP weight matrices") -> Figure:
    layer_count = len(model.layers)
    figure, axes = plt.subplots(1, layer_count, figsize=(4.2 * layer_count, 4.0), squeeze=False)
    axes = axes[0]

    for index, (axis, layer) in enumerate(zip(axes, model.layers, strict=True)):
        matrix = np.asarray(
            [[weight.data for weight in neuron.weights] for neuron in layer.neurons],
            dtype=float,
        )
        max_abs = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            aspect="auto",
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=8)
        axis.set_title(f"Layer {index + 1}")
        axis.set_xlabel("Input unit")
        axis.set_ylabel("Neuron")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    return figure


def plot_mlp_predictions(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ax: Axes | None = None,
    feature_name: str = "x",
    target_name: str = "y",
    title: str = "MLP regression fit",
) -> Figure:
    X = _ensure_2d(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[1] != 1:
        raise ValueError("plot_mlp_predictions supports one input feature.")

    figure, ax = _get_ax(ax)
    x_values = X[:, 0]
    dense_x = np.linspace(x_values.min() - 0.3, x_values.max() + 0.3, 240)
    dense_y = model.predict(dense_x.reshape(-1, 1))
    ax.scatter(x_values, y, color="#0f766e", s=70, edgecolor="white", linewidth=0.9, label="data")
    ax.plot(dense_x, dense_y, color="#7c3aed", linewidth=2.8, label="MLP")
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return figure
