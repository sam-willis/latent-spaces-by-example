import numpy as np

from .types import Array


def adjust_weights_to_positive_orthant(w: Array, *, tolerance: float = 1e-12) -> Array:
    """Clip weights into the positive orthant and renormalize along the last axis."""
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    norms = np.linalg.norm(w, axis=-1, keepdims=True)
    return w / np.maximum(norms, float(tolerance))


def is_on_hypersphere(x: Array, *, tolerance: float = 1e-12) -> bool:
    """Return True if all points have L2 norm ~ 1 along last axis."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1)
    return bool(np.all(np.abs(norms - 1.0) < float(tolerance)))

