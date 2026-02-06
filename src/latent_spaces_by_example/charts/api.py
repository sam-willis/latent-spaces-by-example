from abc import ABC, abstractmethod

from jaxtyping import Float

from ..types import Array


class Chart(ABC):
    """Base class for charts between a hypercube coordinate `u` and spherical weights `w`."""

    def __init__(self, *, tolerance: float = 1e-12) -> None:
        self.tolerance = float(tolerance)

    @abstractmethod
    def forward(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds"]:
        """Map `u` in the unit hypercube ([0,1]^N) to weights `w` on the positive orthant of the unit sphere (S^N_+)."""

    @abstractmethod
    def inverse(
        self,
        w: Float[Array, "n_points n_seeds"],
        *,
        tolerance: float | None = None,
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        """Map weights `w` on the positive orthant of the unit sphere (S^N_+) back to the unit hypercube ([0,1]^N)."""
