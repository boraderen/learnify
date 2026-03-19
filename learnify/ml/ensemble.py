"""Ensemble models built from Learnify primitives."""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Bootstrap aggregation of decision trees."""

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        n_samples = len(X)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._class_to_index = {label: index for index, label in enumerate(self.classes_)}
        self.trees_: list[DecisionTreeClassifier] = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_indices = rng.integers(0, n_samples, size=n_samples)
            else:
                sample_indices = rng.choice(n_samples, size=n_samples, replace=False)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            tree.fit(X[sample_indices], y[sample_indices])
            self.trees_.append(tree)

        self.estimators_ = self.trees_
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.trees_],
            axis=0,
        )
        total = self.feature_importances_.sum()
        if total > 0.0:
            self.feature_importances_ /= total
        return self

    def _aligned_tree_proba(self, tree: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
        raw_probabilities = tree.predict_proba(X)
        aligned = np.zeros((len(X), len(self.classes_)), dtype=float)
        for local_index, label in enumerate(tree.classes_):
            aligned[:, self._class_to_index[label]] = raw_probabilities[:, local_index]
        return aligned

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        probabilities = np.stack(
            [self._aligned_tree_proba(tree, X) for tree in self.trees_],
            axis=0,
        )
        return probabilities.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
