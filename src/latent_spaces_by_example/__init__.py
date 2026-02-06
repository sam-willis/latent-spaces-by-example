"""Latent spaces by example."""

from importlib.metadata import PackageNotFoundError, version

from .surrogate import (
    SurrogateChart,
    angular_surrogate_chart,
    knothe_rosenblatt_surrogate_chart,
    simplex_surrogate_chart,
)

__all__ = [
    "__version__",
    "SurrogateChart",
    "angular_surrogate_chart",
    "knothe_rosenblatt_surrogate_chart",
    "simplex_surrogate_chart",
]

try:
    __version__ = version("latent-spaces-by-example")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g. running from a source checkout without an installed dist).
    __version__ = "0.0.0"
