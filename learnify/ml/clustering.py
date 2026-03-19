"""Clustering algorithms implemented from scratch."""

from __future__ import annotations

import numpy as np


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


class KMeans:
    """A didactic implementation of Lloyd's algorithm."""

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "KMeans":
        X = _ensure_2d(X)
        n_samples = len(X)
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters must be <= the number of samples.")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X, rng)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()

            for cluster_index in range(self.n_clusters):
                members = X[labels == cluster_index]
                if len(members) == 0:
                    new_centers[cluster_index] = X[rng.integers(0, n_samples)]
                else:
                    new_centers[cluster_index] = members.mean(axis=0)

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = np.argmin(
            np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2),
            axis=1,
        )
        self.inertia_ = float(
            np.sum((X - self.cluster_centers_[self.labels_]) ** 2)
        )
        return self

    def _initialize_centers(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Use a small k-means++ style initializer for more stable clusters."""

        n_samples = len(X)
        first_index = int(rng.integers(0, n_samples))
        centers = [X[first_index]]

        while len(centers) < self.n_clusters:
            current_centers = np.asarray(centers)
            distances = np.linalg.norm(X[:, None, :] - current_centers[None, :, :], axis=2) ** 2
            closest_distances = np.min(distances, axis=1)
            total_distance = closest_distances.sum()

            if total_distance == 0.0:
                next_index = int(rng.integers(0, n_samples))
            else:
                probabilities = closest_distances / total_distance
                next_index = int(rng.choice(n_samples, p=probabilities))

            centers.append(X[next_index])

        return np.asarray(centers, dtype=float)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d(X)
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


def _cluster_distance(
    pairwise_distances: np.ndarray,
    left_cluster: list[int],
    right_cluster: list[int],
    linkage: str,
) -> float:
    distances = pairwise_distances[np.ix_(left_cluster, right_cluster)]
    if linkage == "single":
        return float(np.min(distances))
    if linkage == "complete":
        return float(np.max(distances))
    if linkage == "average":
        return float(np.mean(distances))
    raise ValueError("linkage must be one of: single, complete, average")


class AgglomerativeClustering:
    """Naive hierarchical clustering for small educational datasets."""

    def __init__(self, n_clusters: int = 2, linkage: str = "average") -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        X = _ensure_2d(X)
        n_samples = len(X)
        if not 1 <= self.n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and the number of samples.")

        pairwise_distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        clusters = [
            {
                "members": [index],
                "id": index,
            }
            for index in range(n_samples)
        ]
        children: list[tuple[int, int]] = []
        distances: list[float] = []
        next_cluster_id = n_samples

        while len(clusters) > self.n_clusters:
            best_pair = None
            best_distance = float("inf")

            for left_index in range(len(clusters)):
                for right_index in range(left_index + 1, len(clusters)):
                    distance = _cluster_distance(
                        pairwise_distances,
                        clusters[left_index]["members"],
                        clusters[right_index]["members"],
                        self.linkage,
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (left_index, right_index)

            left_index, right_index = best_pair
            left_cluster = clusters[left_index]
            right_cluster = clusters[right_index]
            merged_cluster = {
                "members": left_cluster["members"] + right_cluster["members"],
                "id": next_cluster_id,
            }
            children.append((left_cluster["id"], right_cluster["id"]))
            distances.append(best_distance)
            next_cluster_id += 1
            clusters.pop(right_index)
            clusters.pop(left_index)
            clusters.append(merged_cluster)

        labels = np.empty(n_samples, dtype=int)
        for cluster_index, cluster in enumerate(clusters):
            labels[cluster["members"]] = cluster_index

        self.labels_ = labels
        self.clusters_ = [np.array(cluster["members"], dtype=int) for cluster in clusters]
        self.children_ = np.asarray(children, dtype=int).reshape(-1, 2)
        self.distances_ = np.asarray(distances, dtype=float)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
