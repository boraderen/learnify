# Learnify

Learnify is a didactic, from-scratch Python library for machine learning and deep learning built with raw Python and NumPy. It is designed to stay readable first: the code favors clear implementations of core ideas such as gradient descent, backpropagation, computation graphs, decision trees, ensembles, SVMs, and clustering.

The project is intentionally still being expanded. The current version focuses on a strong educational core that is small enough to study end to end.

## Current capabilities

- Scalar autodiff with reverse-mode backpropagation via a `Value` computation graph
- Computation graph visualization as pure SVG, including each node's value and gradient
- Reusable plotting helpers for linear models, SVMs, trees, forests, clustering, and MLPs
- Gradient-descent models: linear regression and logistic regression
- A small multilayer perceptron built on the autodiff engine
- Classical ML models: decision tree, random forest, and linear SVM
- Clustering algorithms: K-means and agglomerative clustering
- A NumPy-backed RNG utility that stays independent from NumPy's global RNG state

## Install

```bash
uv sync --dev
```

## Quick start

```python
from learnify import Value, computation_graph_svg

x = Value(2.0, label="x")
y = Value(-3.0, label="y")
z = ((x * y) + 4.0).tanh()
z.label = "z"

z.backward()

svg = computation_graph_svg(z)
with open("graph.svg", "w", encoding="utf-8") as handle:
    handle.write(svg)
```

The generated SVG annotates every node with:

- the node label
- `data=<value>`
- `grad=<gradient>`
- the operation that produced the node when relevant

## Example models

```python
import numpy as np
from learnify import DecisionTreeClassifier, KMeans, LinearRegressionGD, MLP

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

regressor = LinearRegressionGD(learning_rate=0.05, n_iterations=1_500)
regressor.fit(X, y)
print(regressor.predict([[4.0]]))

network = MLP(1, [8, 1], seed=0)
history = network.fit(X, y, epochs=80, learning_rate=0.03)
print(history[-1])

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))
print(tree.predict([[0, 1]]))

kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(np.array([[0.0, 0.1], [0.2, -0.1], [5.0, 5.1], [5.2, 4.8]]))
print(labels)
```

## Project layout

- `learnify/core`: autodiff, visualization, optimization utilities, and RNG helpers
- `learnify/nn`: a small neural-network stack built on the computation graph
- `learnify/ml`: classical machine learning algorithms implemented from scratch
- `tests`: lightweight behavioral tests for the educational API surface
