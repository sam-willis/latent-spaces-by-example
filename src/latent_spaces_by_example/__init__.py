"""Latent spaces by example."""

from .surrogate import (
    angular_surrogate_chart,
    knothe_rosenblatt_surrogate_chart,
    make_surrogate_chart,
    simplex_surrogate_chart,
)

__all__ = [
    "__version__",
    "angular_surrogate_chart",
    "knothe_rosenblatt_surrogate_chart",
    "make_surrogate_chart",
    "simplex_surrogate_chart",
]

__version__ = "0.1.0"
