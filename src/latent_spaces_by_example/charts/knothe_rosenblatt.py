import numpy as np
from jaxtyping import Float
from scipy.special import betainc, betaincinv

from ..types import Array
from ..utils import is_on_hypersphere


def knothe_rosenblatt_forward(
    u: Float[Array, "num_points num_dims"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_dims_plus_one"]:
    """Equal-area map from `(0,1)^N` to `S^N_+` using a Knöthe–Rosenblatt construction."""
    tol = float(tolerance)
    if tol <= 0:
        raise ValueError("tolerance must be > 0")

    u = np.asarray(u, dtype=np.float64)
    if not (np.all(u >= -tol) and np.all(u <= 1.0 + tol)):
        raise ValueError("u is not in the unit hypercube (within tolerance)")

    # flip so the first seed is assigned first (matches original code)
    u = 1.0 - u

    _, num_dims = u.shape
    num_dims_plus_one = num_dims + 1

    ks = np.arange(1, num_dims + 1, dtype=np.float64)
    a = 0.5
    b = (num_dims_plus_one - ks) / 2.0

    v = betaincinv(a, b, np.clip(u, 1e-15, 1.0 - 1e-15))  # (..., N)

    z = np.empty(u.shape[:-1] + (num_dims_plus_one,), dtype=np.float64)
    z[..., 0] = v[..., 0]
    tail = 1.0 - v[..., 0]
    for k in range(1, num_dims):
        z[..., k] = tail * v[..., k]
        tail = tail * (1.0 - v[..., k])
    z[..., num_dims] = tail

    x = np.sqrt(np.clip(z, 0.0, 1.0))
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), tol)
    return x


def knothe_rosenblatt_inverse(
    w: Float[Array, "num_points num_dims_plus_one"], *, tolerance: float = 1e-12
) -> Float[Array, "num_points num_dims"]:
    """Inverse of `knothe_rosenblatt_forward` on `S^N_+` (equal-area)."""
    tol = float(tolerance)
    if tol <= 0:
        raise ValueError("tolerance must be > 0")

    w = np.asarray(w, dtype=np.float64)
    _, num_dims_plus_one = w.shape
    if num_dims_plus_one < 2:
        raise ValueError("w must have at least 2 dimensions")
    if not is_on_hypersphere(w, tolerance=tol):
        raise ValueError("w is not on the hypersphere (within tolerance)")
    if not np.all(w >= -tol):
        raise ValueError("w is not in the positive orthant (within tolerance)")

    num_dims = num_dims_plus_one - 1
    z = np.clip(w**2, 0.0, 1.0)

    s = np.flip(np.cumsum(np.flip(z, axis=-1), axis=-1), axis=-1)
    v = z[..., :-1] / np.clip(s[..., :-1], 1e-30, None)

    ks = np.arange(1, num_dims + 1, dtype=np.float64)
    a = 0.5
    b = (num_dims_plus_one - ks) / 2.0
    u = betainc(a, b, np.clip(v, 0.0, 1.0))

    # flip back (matches forward)
    u = 1.0 - u
    return u

