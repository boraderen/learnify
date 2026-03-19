"""Independent RNG helpers built on NumPy's Generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_generator = np.random.default_rng()


def seed(value: int | None) -> None:
    """Reset Learnify's internal generator."""

    global _generator
    _generator = np.random.default_rng(value)


def rand(*shape: int) -> np.ndarray:
    """Return samples from U[0, 1) with NumPy-like shape semantics."""

    if not shape:
        return _generator.random()
    return _generator.random(shape)


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Return normally distributed samples."""

    return _generator.normal(loc=loc, scale=scale, size=size)


def integers(
    low: int,
    high: int | None = None,
    size: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Return random integers in the half-open interval [low, high)."""

    return _generator.integers(low, high=high, size=size)


def choice(
    a: int | Sequence[int] | np.ndarray,
    size: int | Sequence[int] | None = None,
    replace: bool = True,
    p: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Sample from a sequence or integer range."""

    return _generator.choice(a, size=size, replace=replace, p=p)
