"""Linear support vector machine classifier."""

from __future__ import annotations

import numpy as np


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


class LinearSVM:
    """A linear SVM trained with subgradient descent on hinge loss."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        epochs: int = 1_000,
    ) -> None:
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = _ensure_2d(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LinearSVM currently supports binary classification only.")

        signed_targets = np.where(y == self.classes_[0], -1.0, 1.0)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.epochs):
            margins = signed_targets * (X @ self.weights_ + self.bias_)
            active = margins < 1.0
            gradient_w = self.regularization * self.weights_
            gradient_b = 0.0
            if np.any(active):
                gradient_w -= (X[active].T @ signed_targets[active]) / n_samples
                gradient_b -= float(signed_targets[active].sum() / n_samples)

            self.weights_ -= self.learning_rate * gradient_w
            self.bias_ -= self.learning_rate * gradient_b

            loss = (
                0.5 * self.regularization * np.dot(self.weights_, self.weights_)
                + np.maximum(0.0, 1.0 - margins).mean()
            )
            self.loss_history_.append(float(loss))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        decisions = self.decision_function(X)
        return np.where(decisions >= 0.0, self.classes_[1], self.classes_[0])
