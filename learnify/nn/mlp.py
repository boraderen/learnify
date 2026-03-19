"""A tiny multilayer perceptron built on scalar autodiff nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..core.autodiff import Value


def _to_value(value: float | Value) -> Value:
    if isinstance(value, Value):
        return value
    return Value(float(value))


class Module:
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(
        self,
        n_inputs: int,
        *,
        nonlinearity: bool = True,
        generator: np.random.Generator | None = None,
    ) -> None:
        generator = generator or np.random.default_rng()
        self.weights = [Value(float(generator.uniform(-1.0, 1.0))) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.nonlinearity = nonlinearity

    def __call__(self, inputs: Sequence[float | Value]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError("Neuron input width does not match its weights.")
        activation = sum(
            (weight * _to_value(feature) for weight, feature in zip(self.weights, inputs, strict=True)),
            self.bias,
        )
        return activation.tanh() if self.nonlinearity else activation

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]


class Layer(Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        nonlinearity: bool = True,
        generator: np.random.Generator | None = None,
    ) -> None:
        self.neurons = [
            Neuron(n_inputs, nonlinearity=nonlinearity, generator=generator)
            for _ in range(n_outputs)
        ]

    def __call__(self, inputs: Sequence[float | Value]) -> list[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        parameters: list[Value] = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters


class MLP(Module):
    """A simple MLP using tanh hidden activations and MSE training."""

    def __init__(self, n_inputs: int, layer_sizes: Sequence[int], seed: int | None = None) -> None:
        if not layer_sizes:
            raise ValueError("layer_sizes must contain at least one output layer.")

        self.n_inputs_ = n_inputs
        self.layer_sizes_ = list(layer_sizes)
        generator = np.random.default_rng(seed)
        sizes = [n_inputs, *layer_sizes]
        self.layers = [
            Layer(
                sizes[index],
                sizes[index + 1],
                nonlinearity=index < len(layer_sizes) - 1,
                generator=generator,
            )
            for index in range(len(layer_sizes))
        ]

    def __call__(self, inputs: Sequence[float | Value] | float | Value) -> Value | list[Value]:
        if isinstance(inputs, (Value, float, int)):
            activations: list[float | Value] = [inputs]
        else:
            activations = list(inputs)

        for layer in self.layers:
            activations = layer(activations)

        if len(activations) == 1:
            return activations[0]
        return activations

    def parameters(self) -> list[Value]:
        parameters: list[Value] = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 100,
        learning_rate: float = 0.05,
    ) -> list[float]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        history: list[float] = []
        for _ in range(epochs):
            sample_losses: list[Value] = []
            for features, targets in zip(X, y, strict=True):
                prediction = self(features)
                prediction_values = prediction if isinstance(prediction, list) else [prediction]
                losses = [
                    (predicted - float(target)) ** 2
                    for predicted, target in zip(prediction_values, targets, strict=True)
                ]
                sample_losses.append(sum(losses, Value(0.0)) * (1.0 / len(losses)))

            total_loss = sum(sample_losses, Value(0.0)) * (1.0 / len(sample_losses))
            self.zero_grad()
            total_loss.backward()
            for parameter in self.parameters():
                parameter.data -= learning_rate * parameter.grad
            history.append(total_loss.data)

        self.loss_history_ = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions: list[float] | list[list[float]] = []
        for row in X:
            output = self(row)
            if isinstance(output, list):
                predictions.append([float(node.data) for node in output])
            else:
                predictions.append(float(output.data))

        result = np.asarray(predictions, dtype=float)
        if result.ndim == 2 and result.shape[1] == 1:
            return result.reshape(-1)
        return result
