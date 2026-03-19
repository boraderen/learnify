"""Decision tree implementation for didactic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def resolve_max_features(max_features: str | int | float | None, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("Unsupported max_features string.")
    if isinstance(max_features, float):
        if 0.0 < max_features <= 1.0:
            return max(1, int(np.ceil(max_features * n_features)))
        raise ValueError("Float max_features must be in (0, 1].")
    return max(1, min(n_features, int(max_features)))


def _gini(y: np.ndarray, n_classes: int) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


@dataclass(slots=True)
class _TreeNode:
    class_counts: np.ndarray
    predicted_index: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """A small CART-style classifier for numeric features."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: str | int | float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = _ensure_2d(X)
        y = np.asarray(y)
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        self.max_features_ = resolve_max_features(self.max_features, self.n_features_in_)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow_tree(X, encoded_y.astype(int), depth=0)
        self.depth_ = _tree_depth(self.root_)
        self.n_leaves_ = _leaf_count(self.root_)
        self.feature_importances_ = _feature_importances(self.root_, self.n_features_in_)
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        class_counts = np.bincount(y, minlength=self.n_classes_)
        predicted_index = int(np.argmax(class_counts))
        node = _TreeNode(class_counts=class_counts, predicted_index=predicted_index)

        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        is_pure = np.unique(y).size == 1
        too_small = len(y) < self.min_samples_split
        if reached_max_depth or is_pure or too_small:
            return node

        feature_indices = self._sample_feature_indices()
        split = self._best_split(X, y, feature_indices)
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _sample_feature_indices(self) -> np.ndarray:
        if self.max_features_ >= self.n_features_in_:
            return np.arange(self.n_features_in_)
        return self._rng.choice(self.n_features_in_, size=self.max_features_, replace=False)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> tuple[int, float, np.ndarray] | None:
        base_impurity = _gini(y, self.n_classes_)
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        best_left_mask = None

        for feature_index in feature_indices:
            values = X[:, feature_index]
            unique_values = np.unique(values)
            if unique_values.size < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_impurity = _gini(y[left_mask], self.n_classes_)
                right_impurity = _gini(y[right_mask], self.n_classes_)
                weighted_impurity = (
                    left_mask.mean() * left_impurity + right_mask.mean() * right_impurity
                )
                gain = base_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature_index)
                    best_threshold = float(threshold)
                    best_left_mask = left_mask

        if best_feature is None or best_threshold is None or best_left_mask is None:
            return None
        return best_feature, best_threshold, best_left_mask

    def _predict_leaf(self, sample: np.ndarray) -> _TreeNode:
        node = self.root_
        while not node.is_leaf:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        predicted_indices = [self._predict_leaf(sample).predicted_index for sample in X]
        return self.classes_[predicted_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        probabilities = []
        for sample in X:
            leaf = self._predict_leaf(sample)
            probabilities.append(leaf.class_counts / leaf.class_counts.sum())
        return np.asarray(probabilities, dtype=float)


def _node_gini(node: _TreeNode) -> float:
    counts = node.class_counts.astype(float)
    total = counts.sum()
    if total == 0.0:
        return 0.0
    probabilities = counts / total
    return float(1.0 - np.sum(probabilities**2))


def _feature_importances(root: _TreeNode, n_features: int) -> np.ndarray:
    importances = np.zeros(n_features, dtype=float)

    def walk(node: _TreeNode) -> None:
        if node.is_leaf:
            return

        left = node.left
        right = node.right
        node_total = float(node.class_counts.sum())
        left_total = float(left.class_counts.sum())
        right_total = float(right.class_counts.sum())
        improvement = (
            node_total * _node_gini(node)
            - left_total * _node_gini(left)
            - right_total * _node_gini(right)
        )
        importances[node.feature_index] += improvement
        walk(left)
        walk(right)

    walk(root)
    total = importances.sum()
    if total > 0.0:
        importances /= total
    return importances


def _tree_depth(node: _TreeNode) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _leaf_count(node: _TreeNode) -> int:
    if node.is_leaf:
        return 1
    return _leaf_count(node.left) + _leaf_count(node.right)
