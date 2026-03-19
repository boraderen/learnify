import numpy as np

from learnify import (
    AgglomerativeClustering,
    DecisionTreeClassifier,
    KMeans,
    LinearRegressionGD,
    LinearSVM,
    LogisticRegressionGD,
    MLP,
    RandomForestClassifier,
)


def test_linear_regression_learns_simple_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    model = LinearRegressionGD(learning_rate=0.05, n_iterations=2_000)
    model.fit(X, y)

    prediction = model.predict([[4.0]])[0]
    assert abs(prediction - 9.0) < 0.15
    assert model.score(X, y) > 0.99


def test_logistic_regression_separates_binary_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionGD(learning_rate=0.2, n_iterations=2_000)
    model.fit(X, y)

    predictions = model.predict(X)
    assert np.array_equal(predictions, y)


def test_decision_tree_solves_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = DecisionTreeClassifier(max_depth=2, random_state=0)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)


def test_random_forest_handles_simple_quadrants():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [1.0, 2.0],
            [1.5, 1.2],
            [-1.0, 1.5],
            [-1.2, 2.1],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])
    forest = RandomForestClassifier(n_estimators=25, max_depth=4, random_state=0)
    forest.fit(X, y)

    predictions = forest.predict(X)
    assert (predictions == y).mean() >= 0.83


def test_linear_svm_classifies_linearly_separable_points():
    X = np.array([[-2.0, -1.0], [-1.0, -1.5], [1.0, 1.0], [2.0, 1.5]])
    y = np.array([0, 0, 1, 1])
    model = LinearSVM(learning_rate=0.05, regularization=0.01, epochs=2_000)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)


def test_kmeans_finds_two_clusters():
    X = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [5.0, 5.1],
            [5.2, 4.9],
        ]
    )
    model = KMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(X)

    assert len(np.unique(labels)) == 2
    assert model.inertia_ < 0.2


def test_agglomerative_clustering_groups_nearby_points():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [5.0, 5.0],
            [5.2, 5.1],
        ]
    )
    labels = AgglomerativeClustering(n_clusters=2, linkage="average").fit_predict(X)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_mlp_training_reduces_loss_on_small_regression_task():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
    model = MLP(1, [4, 1], seed=0)
    history = model.fit(X, y, epochs=120, learning_rate=0.03)

    assert history[-1] < history[0]
    predictions = model.predict(X)
    assert np.mean((predictions - y) ** 2) < 0.5
