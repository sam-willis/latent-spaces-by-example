import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from .types import Array


@jaxtyped(typechecker=beartype)
def adjust_weights_to_positive_orthant(
    w: Float[Array, "n_points n_seeds"], *, tolerance: float = 1e-12
) -> Float[Array, "n_points n_seeds"]:
    """
    Project weights to the positive orthant of the unit sphere (S^N_+)

    Weights may no longer be valid due to numerical errors, so this reporjects to ensure they are.
    """
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    norms = np.linalg.norm(w, axis=-1, keepdims=True)
    return w / np.maximum(norms, float(tolerance))


@jaxtyped(typechecker=beartype)
def is_on_hypersphere(
    x: Float[Array, "*batch n_dims"], *, tolerance: float = 1e-12
) -> bool:
    """
    Check if all points in a batch have L2 norm ~ 1 along last axis.

    This is used to check if weights are on the unit sphere (S^N_+).
    """
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1)
    return bool(np.all(np.abs(norms - 1.0) < float(tolerance)))
