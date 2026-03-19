"""Common metrics used by Learnify models and tests."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_variance = np.sum((y_true - y_true.mean()) ** 2)
    if total_variance == 0.0:
        return 1.0
    residual_variance = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - residual_variance / total_variance)
