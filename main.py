from pathlib import Path

import numpy as np

from learnify import MLP, Value, computation_graph_svg


def main() -> None:
    x1 = Value(2.0, label="x1")
    x2 = Value(-3.0, label="x2")
    score = ((x1 * x2) + 4.0).tanh()
    score.label = "score"
    score.backward()

    graph_path = Path("computation_graph_demo.svg")
    graph_path.write_text(computation_graph_svg(score), encoding="utf-8")

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
    network = MLP(1, [6, 1], seed=0)
    history = network.fit(X, y, epochs=60, learning_rate=0.03)

    print("Learnify demo")
    print(f"Computation graph written to: {graph_path}")
    print(f"Final MLP loss: {history[-1]:.4f}")


if __name__ == "__main__":
    main()
