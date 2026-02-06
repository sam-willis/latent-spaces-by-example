import numpy as np
import pytest

from latent_spaces_by_example.charts.knothe_rosenblatt import KnotheRosenblattChart


def test_knothe_rosenblatt_raises_on_nonpositive_tolerance() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=(4, 3))
    w = np.ones((4, 4), dtype=np.float64) / 2.0

    chart = KnotheRosenblattChart()
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        chart.forward(u, tolerance=0.0)
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        chart.inverse(w, tolerance=-1.0)


def test_knothe_rosenblatt_forward_raises_on_u_outside_hypercube() -> None:
    u = np.array([[0.5, -0.1]], dtype=np.float64)
    chart = KnotheRosenblattChart(tolerance=1e-12)
    with pytest.raises(ValueError, match="unit hypercube"):
        chart.forward(u, tolerance=1e-6)


def test_knothe_rosenblatt_inverse_raises_on_w_not_on_hypersphere() -> None:
    w = np.array([[1.0, 1.0]], dtype=np.float64)  # norm != 1
    chart = KnotheRosenblattChart(tolerance=1e-12)
    with pytest.raises(ValueError, match="hypersphere"):
        chart.inverse(w, tolerance=1e-12)


def test_knothe_rosenblatt_inverse_raises_on_negative_w() -> None:
    # norm==1 but not in positive orthant
    w = np.array([[-0.1, np.sqrt(1.0 - 0.1**2)]], dtype=np.float64)
    chart = KnotheRosenblattChart(tolerance=1e-12)
    with pytest.raises(ValueError, match="positive orthant"):
        chart.inverse(w, tolerance=1e-12)


def test_knothe_rosenblatt_roundtrip() -> None:
    rng = np.random.default_rng(0)
    n = 64
    d = 4
    u = rng.uniform(size=(n, d))

    chart = KnotheRosenblattChart(tolerance=1e-12)
    w = chart.forward(u)

    assert w.shape == (n, d + 1)
    assert np.all(w >= -1e-12)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-10)

    u2 = chart.inverse(w)
    assert u2.shape == u.shape
    assert np.allclose(u2, u, atol=5e-7)
