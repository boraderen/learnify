import math

from learnify import Value, computation_graph_dot, computation_graph_svg


def test_backward_matches_manual_gradient():
    x = Value(2.0, label="x")
    y = Value(-3.0, label="y")
    z = Value(10.0, label="z")
    q = x * y + z
    out = q.tanh()
    out.label = "out"

    out.backward()

    tanh_prime = 1.0 - math.tanh(q.data) ** 2
    assert abs(x.grad - (y.data * tanh_prime)) < 1e-6
    assert abs(y.grad - (x.data * tanh_prime)) < 1e-6
    assert abs(z.grad - tanh_prime) < 1e-6


def test_computation_graph_visualization_contains_values_and_grads():
    x = Value(1.5, label="x")
    y = (x * 2.0).relu()
    y.label = "y"
    y.backward()

    svg = computation_graph_svg(y)
    dot = computation_graph_dot(y)

    assert "data=" in svg
    assert "grad=" in svg
    assert "x" in svg
    assert "y" in svg
    assert "digraph ComputationGraph" in dot
