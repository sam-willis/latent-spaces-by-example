import numpy as np

from latent_spaces_by_example.charts.knothe_rosenblatt import (
    knothe_rosenblatt_forward,
    knothe_rosenblatt_inverse,
)
from latent_spaces_by_example.surrogate import knothe_rosenblatt_surrogate_chart


def test_knothe_rosenblatt_chart_roundtrip() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=(32, 4))
    w = knothe_rosenblatt_forward(u)

    assert w.shape == (32, 5)
    assert np.all(w >= -1e-12)
    assert np.allclose(np.linalg.norm(w, axis=-1), 1.0, atol=1e-10)

    u2 = knothe_rosenblatt_inverse(w)
    assert u2.shape == u.shape
    assert np.allclose(u2, u, atol=5e-7)


def test_surrogate_chart_shapes() -> None:
    rng = np.random.default_rng(0)
    num_seeds = 6
    latent_dim = 3
    seed_latents = [rng.normal(size=(latent_dim,)) for _ in range(num_seeds)]

    chart = knothe_rosenblatt_surrogate_chart(seed_latents)

    u = rng.uniform(size=(10, num_seeds - 1))
    z = chart["from_u_to_z"](u)
    assert z.shape == (10, latent_dim)

    u2 = chart["from_z_to_u"](z)
    assert u2.shape == u.shape

