import matplotlib

matplotlib.use("Agg")

import io

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
    plotting,
)


def _assert_renderable(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 0


def test_linear_visualizations_render():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    model = LinearRegressionGD(learning_rate=0.05, n_iterations=1_500).fit(X, y)

    _assert_renderable(plotting.plot_regression_fit(model, X, y))
    _assert_renderable(plotting.plot_linear_coefficients(model))
    _assert_renderable(plotting.plot_loss_curve(model.loss_history_))


def test_binary_classification_visualizations_render():
    X = np.array([[-2.0, -1.0], [-1.0, -1.5], [1.0, 1.0], [2.0, 1.5]])
    y = np.array([0, 0, 1, 1])

    logistic = LogisticRegressionGD(learning_rate=0.2, n_iterations=1_200).fit(X, y)
    svm = LinearSVM(learning_rate=0.05, regularization=0.01, epochs=1_200).fit(X, y)

    _assert_renderable(plotting.plot_binary_decision_surface(logistic, X, y, response="probability"))
    _assert_renderable(plotting.plot_discriminant_function(logistic, X, y))
    _assert_renderable(plotting.plot_binary_decision_surface(svm, X, y, response="decision"))
    _assert_renderable(plotting.plot_discriminant_function(svm, X, y))


def test_tree_and_forest_visualizations_render():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -0.3],
            [1.0, 2.0],
            [1.4, 1.5],
            [-1.0, 1.5],
            [-1.4, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    forest = RandomForestClassifier(n_estimators=6, max_depth=3, random_state=0).fit(X, y)

    assert np.isclose(tree.feature_importances_.sum(), 1.0)
    assert np.isclose(forest.feature_importances_.sum(), 1.0)
    _assert_renderable(plotting.plot_decision_tree(tree))
    _assert_renderable(plotting.plot_tree_feature_importances(tree))
    _assert_renderable(plotting.plot_random_forest(forest, max_trees=3))


def test_clustering_visualizations_render():
    X = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.1, 0.0],
            [5.0, 5.1],
            [5.2, 4.9],
            [4.9, 5.2],
        ]
    )
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    agg = AgglomerativeClustering(n_clusters=2, linkage="average").fit(X)

    assert agg.children_.shape[1] == 2
    _assert_renderable(plotting.plot_kmeans_clusters(kmeans, X))
    _assert_renderable(plotting.plot_agglomerative_clusters(agg, X))
    _assert_renderable(plotting.plot_agglomerative_dendrogram(agg))


def test_mlp_visualizations_render():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
    model = MLP(1, [4, 1], seed=0)
    model.fit(X, y, epochs=60, learning_rate=0.03)

    _assert_renderable(plotting.plot_mlp_architecture(model))
    _assert_renderable(plotting.plot_mlp_weight_matrices(model))
    _assert_renderable(plotting.plot_mlp_predictions(model, X, y))
