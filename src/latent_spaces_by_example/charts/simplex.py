import numpy as np
from jaxtyping import Float

from ..types import Array


def simplex_chart_forward(
    u: Float[Array, "num_points num_seeds_minus_one"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_seeds"]:
    """Map u in [0,1]^D to w on the positive-orthant unit sphere using a simplex-inspired chart."""
    u = np.asarray(u, dtype=np.float64)
    v_first = 1.0 - u  # (N, D)
    v_last = np.sum(u, axis=-1, keepdims=True)  # (N, 1)
    v = np.concatenate([v_first, v_last], axis=-1)  # (N, D+1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, float(tolerance))


def simplex_chart_inverse(
    w: Float[Array, "num_points num_seeds"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_seeds_minus_one"]:
    """Inverse of `simplex_chart_forward` on its image (renormalizes defensively)."""
    w = np.asarray(w, dtype=np.float64)
    w = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), float(tolerance))

    d = w.shape[-1] - 1
    s = np.sum(w, axis=-1, keepdims=True)  # (N, 1)
    t = d / np.maximum(s, float(tolerance))
    u = 1.0 - t * w[..., :d]
    return np.clip(u, 0.0, 1.0)

