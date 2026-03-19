"""Gradient-descent linear models."""

from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class LinearRegressionGD:
    """Linear regression optimized with full-batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1_000,
        fit_intercept: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.n_iterations):
            predictions = X @ self.weights_ + self.bias_
            errors = predictions - y
            gradient_w = (2.0 / n_samples) * (X.T @ errors)
            gradient_b = 2.0 * float(errors.mean())
            self.weights_ -= self.learning_rate * gradient_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * gradient_b
            self.loss_history_.append(float(np.mean(errors**2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        return X @ self.weights_ + self.bias_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


class LogisticRegressionGD:
    """Binary logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1_000,
        l2: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = _ensure_2d(X)
        y = np.asarray(y)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegressionGD currently supports binary classification only.")

        y_encoded = y_encoded.astype(float)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.n_iterations):
            logits = X @ self.weights_ + self.bias_
            probabilities = _sigmoid(logits)
            errors = probabilities - y_encoded
            gradient_w = (X.T @ errors) / n_samples + self.l2 * self.weights_
            gradient_b = float(errors.mean())
            self.weights_ -= self.learning_rate * gradient_w
            self.bias_ -= self.learning_rate * gradient_b

            clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
            loss = (
                -np.mean(y_encoded * np.log(clipped) + (1.0 - y_encoded) * np.log(1.0 - clipped))
                + 0.5 * self.l2 * np.dot(self.weights_, self.weights_)
            )
            self.loss_history_.append(float(loss))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        return X @ self.weights_ + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        positive = self.predict_proba(X)[:, 1] >= threshold
        return self.classes_[positive.astype(int)]
