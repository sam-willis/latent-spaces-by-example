import numpy as np

from latent_spaces_by_example.charts.angular import AngularChart


def test_angular_forward_invariants() -> None:
    rng = np.random.default_rng(0)
    n = 64
    d = 4
    u = rng.uniform(size=(n, d))

    chart = AngularChart(tolerance=1e-12)
    w = chart.forward(u)

    assert w.shape == (n, d + 1)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-10)


def test_angular_inverse_forward_roundtrip_for_interior_u() -> None:
    rng = np.random.default_rng(1)
    n = 128
    d = 3
    # Stay away from endpoints since forward/inverse apply tolerance clipping.
    u = rng.uniform(low=0.05, high=0.95, size=(n, d))

    chart = AngularChart(tolerance=1e-12)
    u2 = chart.inverse(chart.forward(u))

    assert u2.shape == u.shape
    assert np.allclose(u2, u, atol=5e-8)


def test_angular_inverse_d0_returns_empty() -> None:
    # For a 1D weight vector (d=0), inverse should return u with shape (n, 0).
    w = np.array([[1.0], [0.5]], dtype=np.float64)
    chart = AngularChart(tolerance=1e-12)
    u = chart.inverse(w)

    assert u.shape == (2, 0)


def test_angular_forward_clips_and_is_finite_for_out_of_range_u() -> None:
    # AngularChart.forward does not validate u; it clips it.
    u = np.array([[-1.0, 2.0, 0.5]], dtype=np.float64)
    chart = AngularChart(tolerance=1e-12)
    w = chart.forward(u)

    assert np.all(np.isfinite(w))
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)
