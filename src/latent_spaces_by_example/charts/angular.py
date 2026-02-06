import numpy as np
from jaxtyping import Float

from ..types import Array
from .api import Chart


class AngularChart(Chart):
    def forward(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds"]:
        tol = self.tolerance if tolerance is None else float(tolerance)
        u = np.asarray(u, dtype=np.float64)
        u = np.clip(u, tol, 1.0 - tol)
        n, _d = u.shape

        theta = 0.5 * np.pi * u
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        sin_cum = np.cumprod(sin_t, axis=-1)
        lead = np.ones((n, 1), dtype=np.float64)
        sin_prefix = np.concatenate([lead, sin_cum[:, :-1]], axis=-1)

        w_first_d = sin_prefix * cos_t
        w_last = sin_cum[:, -1:]
        w = np.concatenate([w_first_d, w_last], axis=-1)

        norms = np.linalg.norm(w, axis=-1, keepdims=True)
        w = w / np.maximum(norms, tol)
        return np.clip(w, 0.0, 1.0)

    def inverse(
        self,
        w: Float[Array, "n_points n_seeds"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        tol = self.tolerance if tolerance is None else float(tolerance)

        w = np.asarray(w, dtype=np.float64)
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


def angular_chart_forward(
    u: Float[Array, "n_points n_seeds_minus_one"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds"]:
    """Functional wrapper for :meth:`AngularChart.forward`."""
    return AngularChart(tolerance=float(tolerance)).forward(
        u, tolerance=float(tolerance)
    )


def angular_chart_inverse(
    w: Float[Array, "n_points n_seeds"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds_minus_one"]:
    """Functional wrapper for :meth:`AngularChart.inverse`."""
    return AngularChart(tolerance=float(tolerance)).inverse(
        w, tolerance=float(tolerance)
    )
