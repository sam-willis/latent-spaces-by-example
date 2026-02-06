import numpy as np
from jaxtyping import Float
from scipy.special import betainc, betaincinv

from ..types import Array
from ..utils import is_on_hypersphere
from .api import Chart


class KnotheRosenblattChart(Chart):
    def forward(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds"]:
        tol = self.tolerance if tolerance is None else float(tolerance)
        if tol <= 0:
            raise ValueError("tolerance must be > 0")

        u = np.asarray(u, dtype=np.float64)
        if not (np.all(u >= -tol) and np.all(u <= 1.0 + tol)):
            raise ValueError("u is not in the unit hypercube (within tolerance)")

        u = 1.0 - u
        _, n_dims = u.shape
        n_dims_plus_one = n_dims + 1

        ks = np.arange(1, n_dims + 1, dtype=np.float64)
        a = 0.5
        b = (n_dims_plus_one - ks) / 2.0

        v = betaincinv(a, b, np.clip(u, 1e-15, 1.0 - 1e-15))

        z = np.empty(u.shape[:-1] + (n_dims_plus_one,), dtype=np.float64)
        z[..., 0] = v[..., 0]
        tail = 1.0 - v[..., 0]
        for k in range(1, n_dims):
            z[..., k] = tail * v[..., k]
            tail = tail * (1.0 - v[..., k])
        z[..., n_dims] = tail

        x = np.sqrt(np.clip(z, 0.0, 1.0))
        x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), tol)
        return x

    def inverse(
        self,
        w: Float[Array, "n_points n_seeds"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        tol = self.tolerance if tolerance is None else float(tolerance)
        if tol <= 0:
            raise ValueError("tolerance must be > 0")

        w = np.asarray(w, dtype=np.float64)
        _, n_dims_plus_one = w.shape
        if not is_on_hypersphere(w, tolerance=tol):
            raise ValueError("w is not on the hypersphere (within tolerance)")
        if not np.all(w >= -tol):
            raise ValueError("w is not in the positive orthant (within tolerance)")

        n_dims = n_dims_plus_one - 1
        z = np.clip(w**2, 0.0, 1.0)

        s = np.flip(np.cumsum(np.flip(z, axis=-1), axis=-1), axis=-1)
        v = z[..., :-1] / np.clip(s[..., :-1], 1e-30, None)

        ks = np.arange(1, n_dims + 1, dtype=np.float64)
        a = 0.5
        b = (n_dims_plus_one - ks) / 2.0
        u = betainc(a, b, np.clip(v, 0.0, 1.0))

        u = 1.0 - u
        return u


def knothe_rosenblatt_forward(
    u: Float[Array, "n_points n_seeds_minus_one"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds"]:
    """Functional wrapper for :meth:`KnotheRosenblattChart.forward`."""
    return KnotheRosenblattChart(tolerance=float(tolerance)).forward(
        u, tolerance=float(tolerance)
    )


def knothe_rosenblatt_inverse(
    w: Float[Array, "n_points n_seeds"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds_minus_one"]:
    """Functional wrapper for :meth:`KnotheRosenblattChart.inverse`."""
    return KnotheRosenblattChart(tolerance=float(tolerance)).inverse(
        w, tolerance=float(tolerance)
    )
