"""Scalar reverse-mode autodiff primitives."""

from __future__ import annotations

import math

Number = float | int


def _ensure_value(value: Number | "Value") -> "Value":
    if isinstance(value, Value):
        return value
    return Value(float(value))


class Value:
    """A scalar node in a computation graph."""

    def __init__(
        self,
        data: Number,
        _children: tuple["Value", ...] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return (
            f"Value(data={self.data:.6f}, grad={self.grad:.6f}, "
            f"label={self.label!r}, op={self._op!r})"
        )

    def __hash__(self) -> int:
        return id(self)

    def __float__(self) -> float:
        return self.data

    def __add__(self, other: Number | "Value") -> "Value":
        other_value = _ensure_value(other)
        out = Value(self.data + other_value.data, (self, other_value), op="+")

        def _backward() -> None:
            self.grad += out.grad
            other_value.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Number | "Value") -> "Value":
        return self + other

    def __mul__(self, other: Number | "Value") -> "Value":
        other_value = _ensure_value(other)
        out = Value(self.data * other_value.data, (self, other_value), op="*")

        def _backward() -> None:
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Number | "Value") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: Number | "Value") -> "Value":
        return self + (-_ensure_value(other))

    def __rsub__(self, other: Number | "Value") -> "Value":
        return _ensure_value(other) + (-self)

    def __truediv__(self, other: Number | "Value") -> "Value":
        return self * (_ensure_value(other) ** -1)

    def __rtruediv__(self, other: Number | "Value") -> "Value":
        return _ensure_value(other) * (self ** -1)

    def __pow__(self, exponent: Number) -> "Value":
        if isinstance(exponent, Value):
            raise TypeError("Value exponents are not supported in this didactic engine.")

        out = Value(self.data**float(exponent), (self,), op=f"**{exponent}")

        def _backward() -> None:
            self.grad += float(exponent) * (self.data ** (float(exponent) - 1.0)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), op="exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        tanh_value = math.tanh(self.data)
        out = Value(tanh_value, (self,), op="tanh")

        def _backward() -> None:
            self.grad += (1.0 - tanh_value**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), op="relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        if self.data >= 0.0:
            exp_neg = math.exp(-self.data)
            sigma = 1.0 / (1.0 + exp_neg)
        else:
            exp_pos = math.exp(self.data)
            sigma = exp_pos / (1.0 + exp_pos)
        out = Value(sigma, (self,), op="sigmoid")

        def _backward() -> None:
            self.grad += sigma * (1.0 - sigma) * out.grad

        out._backward = _backward
        return out

    def topo(self) -> list["Value"]:
        ordered: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            ordered.append(node)

        build(self)
        return ordered

    def zero_grad(self) -> None:
        for node in self.topo():
            node.grad = 0.0

    def backward(self, grad: float = 1.0) -> None:
        ordered = self.topo()
        for node in ordered:
            node.grad = 0.0
        self.grad = grad
        for node in reversed(ordered):
            node._backward()
