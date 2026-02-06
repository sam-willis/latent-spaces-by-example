import numpy as np
from jaxtyping import Float

from ..types import Array
from .api import Chart


class SimplexChart(Chart):
    def forward(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds"]:
        tol = self.tolerance if tolerance is None else float(tolerance)
        u = np.asarray(u, dtype=np.float64)
        v_first = 1.0 - u  # (N, D)
        v_last = np.sum(u, axis=-1, keepdims=True)  # (N, 1)
        v = np.concatenate([v_first, v_last], axis=-1)  # (N, D+1)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, tol)

    def inverse(
        self,
        w: Float[Array, "n_points n_seeds"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        tol = self.tolerance if tolerance is None else float(tolerance)
        w = np.asarray(w, dtype=np.float64)
        w = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), tol)

        d = w.shape[-1] - 1
        s = np.sum(w, axis=-1, keepdims=True)
        t = d / np.maximum(s, tol)
        u = 1.0 - t * w[..., :d]
        return np.clip(u, 0.0, 1.0)


def simplex_chart_forward(
    u: Float[Array, "n_points n_seeds_minus_one"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds"]:
    """Functional wrapper for :meth:`SimplexChart.forward`."""
    return SimplexChart(tolerance=float(tolerance)).forward(
        u, tolerance=float(tolerance)
    )


def simplex_chart_inverse(
    w: Float[Array, "n_points n_seeds"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds_minus_one"]:
    """Functional wrapper for :meth:`SimplexChart.inverse`."""
    return SimplexChart(tolerance=float(tolerance)).inverse(
        w, tolerance=float(tolerance)
    )
