import numpy as np
from jaxtyping import Float

from ..types import Array


def angular_chart_forward(
    u: Float[Array, "num_points num_dims"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_dims_plus_one"]:
    """Map `(0,1)^D -> S^D_+` (positive orthant of unit sphere) via spherical angles."""
    u = np.asarray(u, dtype=np.float64)
    tol = float(tolerance)
    u = np.clip(u, tol, 1.0 - tol)
    n, d = u.shape

    theta = 0.5 * np.pi * u  # (N, D)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    sin_cum = np.cumprod(sin_t, axis=-1)  # (N, D)

    lead = np.ones((n, 1), dtype=np.float64)
    sin_prefix = np.concatenate([lead, sin_cum[:, :-1]], axis=-1)  # (N, D)

    w_first_d = sin_prefix * cos_t  # (N, D)
    w_last = sin_cum[:, -1:]  # (N, 1)
    w = np.concatenate([w_first_d, w_last], axis=-1)  # (N, D+1)

    norms = np.linalg.norm(w, axis=-1, keepdims=True)
    w = w / np.maximum(norms, tol)
    return np.clip(w, 0.0, 1.0)


def angular_chart_inverse(
    z: Float[Array, "num_points num_dims_plus_one"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_dims"]:
    """Inverse chart `S^D_+ -> (0,1)^D` (renormalizes and clamps for stability)."""
    w = np.asarray(z, dtype=np.float64)
    tol = float(tolerance)

    norms = np.linalg.norm(w, axis=-1, keepdims=True)
    w = w / np.maximum(norms, tol)
    w = np.clip(w, 0.0, 1.0)

    n, dp1 = w.shape
    d = dp1 - 1
    if d == 0:
        return np.empty((n, 0), dtype=np.float64)

    theta = np.empty((n, d), dtype=np.float64)
    t1 = np.arccos(np.clip(w[:, 0], 0.0, 1.0))
    theta[:, 0] = t1
    sprod = np.sin(t1)

    for k in range(1, d):
        denom = np.maximum(sprod, tol)
        arg = np.clip(w[:, k] / denom, 0.0, 1.0)
        tk = np.arccos(arg)
        theta[:, k] = tk
        sprod = sprod * np.sin(tk)

    u = (2.0 / np.pi) * theta
    return np.clip(u, tol, 1.0 - tol)

