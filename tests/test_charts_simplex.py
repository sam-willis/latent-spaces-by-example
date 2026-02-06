import numpy as np

from latent_spaces_by_example.charts.simplex import SimplexChart


def test_simplex_forward_invariants() -> None:
    rng = np.random.default_rng(0)
    n = 64
    d = 5
    u = rng.uniform(size=(n, d))

    chart = SimplexChart(tolerance=1e-12)
    w = chart.forward(u)

    assert w.shape == (n, d + 1)
    assert np.all(w >= 0.0)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-12)


def test_simplex_inverse_forward_roundtrip() -> None:
    rng = np.random.default_rng(1)
    n = 128
    d = 4
    u = rng.uniform(size=(n, d))

    chart = SimplexChart(tolerance=1e-12)
    u2 = chart.inverse(chart.forward(u))

    assert u2.shape == u.shape
    assert np.allclose(u2, u, atol=1e-12)


def test_simplex_inverse_d0_returns_empty() -> None:
    w = np.array([[1.0], [0.1]], dtype=np.float64)
    chart = SimplexChart(tolerance=1e-12)
    u = chart.inverse(w)

    assert u.shape == (2, 0)


def test_simplex_inverse_clips_to_unit_hypercube() -> None:
    # Inverse does not enforce positivity; it normalizes and clips the resulting u to [0, 1].
    w = np.array([[-1.0, 0.0, 0.0]], dtype=np.float64)
    chart = SimplexChart(tolerance=1e-12)
    u = chart.inverse(w)

    assert np.all(np.isfinite(u))
    assert np.all(u >= 0.0)
    assert np.all(u <= 1.0)
