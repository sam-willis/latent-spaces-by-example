from abc import ABC, abstractmethod

from beartype import beartype
from jaxtyping import Float, jaxtyped

from ..types import Array


class TransportMap(ABC):
    """Base class for batched transport maps.

    A transport map converts latents representations from one distribution to
    an inner latent space which is amenable to the surrogate chart
    (i.e. uniform on the unit sphere, unit gaussian etc.).
    """

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def forward(
        self, z: Float[Array, "n_points n_latent_dims"]
    ) -> Float[Array, "n_points n_inner_latent_dims"]:
        """Map latents to inner latent space"""

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def backward(
        self, xi: Float[Array, "n_points n_inner_latent_dims"]
    ) -> Float[Array, "n_points n_latent_dims"]:
        """Map inner latents to latent space"""


class IdentityTransportMap(TransportMap):
    """
    Transport map which does not change the latent space,
    i.e. the latent space is already amenable to the surrogate chart.
    (e.g. use if the latent space is already uniform on the unit sphere, unit gaussian etc.).
    """

    @jaxtyped(typechecker=beartype)
    def forward(
        self, z: Float[Array, "n_points n_latent_dims"]
    ) -> Float[Array, "n_points n_inner_latent_dims"]:
        xi = z
        return xi

    @jaxtyped(typechecker=beartype)
    def backward(
        self, xi: Float[Array, "n_points n_inner_latent_dims"]
    ) -> Float[Array, "n_points n_latent_dims"]:
        z = xi
        return z
