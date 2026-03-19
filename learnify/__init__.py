"""Learnify public API."""

from . import plotting
from .core import rng
from .core.autodiff import Value
from .core.optim import GradientDescentOptimizer, gradient_descent_step
from .core.visualization import (
    computation_graph_dot,
    computation_graph_svg,
    save_computation_graph_svg,
    trace_graph,
)
from .metrics import accuracy_score, mean_squared_error, r2_score
from .ml import (
    AgglomerativeClustering,
    DecisionTreeClassifier,
    KMeans,
    LinearRegressionGD,
    LinearSVM,
    LogisticRegressionGD,
    RandomForestClassifier,
)
from .nn import MLP

__all__ = [
    "AgglomerativeClustering",
    "DecisionTreeClassifier",
    "GradientDescentOptimizer",
    "KMeans",
    "LinearRegressionGD",
    "LinearSVM",
    "LogisticRegressionGD",
    "MLP",
    "RandomForestClassifier",
    "Value",
    "accuracy_score",
    "computation_graph_dot",
    "computation_graph_svg",
    "gradient_descent_step",
    "mean_squared_error",
    "plotting",
    "r2_score",
    "rng",
    "save_computation_graph_svg",
    "trace_graph",
]
