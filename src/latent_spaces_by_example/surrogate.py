import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from .charts.angular import AngularChart
from .charts.api import Chart
from .charts.knothe_rosenblatt import KnotheRosenblattChart
from .transport_maps import IdentityTransportMap, TransportMap
from .types import Array
from .utils import adjust_weights_to_positive_orthant, is_on_hypersphere


@jaxtyped(typechecker=beartype)
def from_w_to_inner_latent(
    w: Float[Array, "n_points n_seeds"],
    inner_seed_latents: Float[Array, "n_seeds n_inner_latent_dims"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_inner_latent_dims"]:
    if not is_on_hypersphere(w, tolerance=tolerance):
        raise ValueError("w is not on the hypersphere (within tolerance)")
    return np.matmul(w, inner_seed_latents)


@jaxtyped(typechecker=beartype)
def from_inner_latent_to_w(
    inner_latent: Float[Array, "n_points n_inner_latent_dims"],
    inner_seed_latents_pseudoinverse: Float[Array, "n_inner_latent_dims n_seeds"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "n_points n_seeds"]:
    w_not_projected = np.matmul(inner_latent, inner_seed_latents_pseudoinverse)
    norms = np.linalg.norm(w_not_projected, axis=-1, keepdims=True)
    return w_not_projected / np.maximum(norms, float(tolerance))


class SurrogateChart:
    """Surrogate chart mapping between:

    - `u`: coordinates in a unit hypercube, shape (N, num_seeds - 1)
    - `w`: nonnegative weights on the unit sphere, shape (N, num_seeds)
    - `z`: latent vectors, shape (N, num_latent_dims)

    The surrogate chart is defined by:
    - `weight_chart`: a chart mapping from `u` to `w`
    - `transport_map`: a transport map mapping from `z` to `inner_latent`.

    In many cases the transport map can be omitted, in which case the surrogate chart is simply the identity map.
    This is valid when 'z' are already distributed according to a valid distribution (i.e. uniform on sphere, unit gaussian etc.)
    """

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        seed_latents: Float[Array, "n_seeds n_latent_dims"],
        *,
        weight_chart: Chart | None = None,
        transport_map: TransportMap | None = None,
        tolerance: float = 1e-12,
    ) -> None:
        self._tol = float(tolerance)
        self._weight_chart = weight_chart or KnotheRosenblattChart(tolerance=self._tol)
        self._transport_map = transport_map or IdentityTransportMap()

        inner_seed_latents = self._transport_map.forward(seed_latents)

        self.num_seeds = len(seed_latents)
        self.num_latent_dims = inner_seed_latents.shape[1]
        self._inner_seed_latents = inner_seed_latents
        self._inner_seed_latents_pinv = np.linalg.pinv(inner_seed_latents)

    @jaxtyped(typechecker=beartype)
    def from_u_to_w(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
    ) -> Float[Array, "n_points n_seeds"]:
        return self._weight_chart.forward(u, tolerance=self._tol)

    @jaxtyped(typechecker=beartype)
    def from_w_to_u(
        self,
        w: Float[Array, "n_points n_seeds"],
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        return self._weight_chart.inverse(w, tolerance=self._tol)

    @jaxtyped(typechecker=beartype)
    def from_w_to_z(
        self,
        w: Float[Array, "n_points n_seeds"],
    ) -> Float[Array, "n_points n_latent_dims"]:
        inner = from_w_to_inner_latent(w, self._inner_seed_latents, tolerance=self._tol)
        return self._transport_map.backward(inner)

    @jaxtyped(typechecker=beartype)
    def from_z_to_w(
        self,
        z: Float[Array, "n_points n_latent_dims"],
    ) -> Float[Array, "n_points n_seeds"]:
        inner_latent = self._transport_map.forward(z)
        w = from_inner_latent_to_w(
            inner_latent=inner_latent,
            inner_seed_latents_pseudoinverse=self._inner_seed_latents_pinv,
            tolerance=self._tol,
        )
        return adjust_weights_to_positive_orthant(w, tolerance=self._tol)

    @jaxtyped(typechecker=beartype)
    def from_u_to_z(
        self,
        u: Float[Array, "n_points n_seeds_minus_one"],
    ) -> Float[Array, "n_points n_latent_dims"]:
        return self.from_w_to_z(self.from_u_to_w(u))

    @jaxtyped(typechecker=beartype)
    def from_z_to_u(
        self,
        z: Float[Array, "n_points n_latent_dims"],
    ) -> Float[Array, "n_points n_seeds_minus_one"]:
        return self.from_w_to_u(self.from_z_to_w(z))


@jaxtyped(typechecker=beartype)
def angular_surrogate_chart(
    seed_latents: Float[Array, "n_seeds n_latent_dims"],
    *,
    transport_map: TransportMap | None = None,
) -> SurrogateChart:
    return SurrogateChart(
        seed_latents=seed_latents,
        weight_chart=AngularChart(),
        transport_map=transport_map,
    )


@jaxtyped(typechecker=beartype)
def knothe_rosenblatt_surrogate_chart(
    seed_latents: Float[Array, "n_seeds n_latent_dims"],
    *,
    transport_map: TransportMap | None = None,
) -> SurrogateChart:
    return SurrogateChart(
        seed_latents=seed_latents,
        weight_chart=KnotheRosenblattChart(),
        transport_map=transport_map,
    )
