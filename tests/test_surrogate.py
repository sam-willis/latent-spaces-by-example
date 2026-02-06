import numpy as np
import pytest

from latent_spaces_by_example.surrogate import (
    SurrogateChart,
    from_inner_latent_to_w,
    from_w_to_inner_latent,
)


def test_from_w_to_inner_latent_raises_if_not_on_hypersphere() -> None:
    w = np.array([[2.0, 0.0]], dtype=np.float64)  # norm=2
    inner_seed_latents = np.eye(2, dtype=np.float64)

    with pytest.raises(ValueError, match="hypersphere"):
        from_w_to_inner_latent(w, inner_seed_latents, tolerance=1e-12)


def test_from_inner_latent_to_w_normalizes_nonzero_rows() -> None:
    rng = np.random.default_rng(0)
    n_points = 16
    n_seeds = 5
    n_inner = 3

    inner_seed_latents = rng.normal(size=(n_seeds, n_inner))
    pinv = np.linalg.pinv(inner_seed_latents)

    inner_latent = rng.normal(size=(n_points, n_inner))
    w = from_inner_latent_to_w(inner_latent, pinv, tolerance=1e-12)

    assert w.shape == (n_points, n_seeds)
    assert np.all(np.isfinite(w))
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-10)


def test_from_inner_latent_to_w_all_zero_row_is_finite() -> None:
    n_points = 2
    n_seeds = 4
    n_inner = 3

    inner_latent = np.zeros((n_points, n_inner), dtype=np.float64)
    pinv = np.zeros((n_inner, n_seeds), dtype=np.float64)

    w = from_inner_latent_to_w(inner_latent, pinv, tolerance=1e-12)
    assert np.all(np.isfinite(w))
    assert np.all(w == 0.0)


def test_surrogatechart_from_z_to_w_nonnegative_and_on_hypersphere_for_positive_z() -> (
    None
):
    # Choose seed_latents such that the pseudoinverse is exactly identity.
    seed_latents = np.eye(3, dtype=np.float64)
    chart = SurrogateChart(seed_latents, tolerance=1e-12)

    rng = np.random.default_rng(0)
    z = rng.uniform(low=0.1, high=1.0, size=(32, 3))
    w = chart.from_z_to_w(z)

    assert w.shape == (32, 3)
    assert np.all(w >= 0.0)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-10)
