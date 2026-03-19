"""Optimization helpers for both NumPy arrays and Value parameters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .autodiff import Value


def gradient_descent_step(
    parameters: Iterable[np.ndarray],
    gradients: Iterable[np.ndarray],
    learning_rate: float,
) -> None:
    """Apply one in-place gradient descent step to NumPy parameters."""

    for parameter, gradient in zip(parameters, gradients, strict=False):
        parameter -= learning_rate * gradient


class GradientDescentOptimizer:
    """Minimal optimizer for educational experiments."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step_values(self, parameters: Iterable[Value]) -> None:
        for parameter in parameters:
            parameter.data -= self.learning_rate * parameter.grad

    def zero_grad(self, parameters: Iterable[Value]) -> None:
        for parameter in parameters:
            parameter.grad = 0.0

    def step_arrays(
        self,
        parameters: Iterable[np.ndarray],
        gradients: Iterable[np.ndarray],
    ) -> None:
        gradient_descent_step(parameters, gradients, self.learning_rate)
