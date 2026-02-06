from collections.abc import Callable
from typing import Any

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from .charts.angular import angular_chart_forward, angular_chart_inverse
from .charts.knothe_rosenblatt import knothe_rosenblatt_forward, knothe_rosenblatt_inverse
from .charts.simplex import simplex_chart_forward, simplex_chart_inverse
from .types import Array
from .utils import adjust_weights_to_positive_orthant, is_on_hypersphere


@jaxtyped(typechecker=beartype)
def from_w_to_inner_latent(
    w: Float[Array, "num_points num_seeds"],
    inner_seed_latents: Float[Array, "num_seeds num_latent_dims"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "num_points num_latent_dims"]:
    if not is_on_hypersphere(w, tolerance=tolerance):
        raise ValueError("w is not on the hypersphere (within tolerance)")
    return np.matmul(w, inner_seed_latents)


@jaxtyped(typechecker=beartype)
def from_inner_latent_to_w(
    inner_latent: Float[Array, "num_points num_latent_dims"],
    inner_seed_latents_pseudoinverse: Float[Array, "num_latent_dims num_seeds"],
    *,
    tolerance: float = 1e-12,
) -> Float[Array, "num_points num_seeds"]:
    w_not_projected = np.matmul(inner_latent, inner_seed_latents_pseudoinverse)
    norms = np.linalg.norm(w_not_projected, axis=-1, keepdims=True)
    return w_not_projected / np.maximum(norms, float(tolerance))


def make_surrogate_chart(
    seed_latents: list[Any],
    *,
    weight_chart_forward: Callable[
        [Float[Array, "num_points num_seeds_minus_one"]],
        Float[Array, "num_points num_seeds"],
    ] = knothe_rosenblatt_forward,
    weight_chart_inverse: Callable[
        [Float[Array, "num_points num_seeds"]],
        Float[Array, "num_points num_seeds_minus_one"],
    ] = knothe_rosenblatt_inverse,
    transport_map_forward: Callable[[Any], Float[Array, "num_latent_dims"]] = lambda x: x,
    transport_map_backward: Callable[[Float[Array, "num_latent_dims"]], Any] = lambda x: x,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Build a function-only surrogate chart API (no classes).

    Returns a dict containing closures like:
      - from_u_to_z(u), from_z_to_u(z)
      - from_u_to_w(u), from_w_to_u(w)
      - from_w_to_z(w), from_z_to_w(z)
    plus metadata: num_seeds, num_latent_dims.
    """
    inner_seed_latents = np.stack(
        [transport_map_forward(latent) for latent in seed_latents],
        axis=0,
    )
    if inner_seed_latents.ndim != 2:
        raise ValueError(
            "transport_map_forward must produce vectors; stacking must form a rank-2 array"
        )

    num_seeds, num_latent_dims = inner_seed_latents.shape
    if num_seeds < 2:
        raise ValueError("num_seeds must be at least 2")
    if num_latent_dims <= 0:
        raise ValueError("num_latent_dims must be positive")

    inner_seed_latents_pinv = np.linalg.pinv(inner_seed_latents)
    tol = float(tolerance)

    @jaxtyped(typechecker=beartype)
    def from_u_to_w(
        u: Float[Array, "num_points num_seeds_minus_one"],
    ) -> Float[Array, "num_points num_seeds"]:
        return weight_chart_forward(u, tolerance=tol)  # type: ignore[call-arg]

    @jaxtyped(typechecker=beartype)
    def from_w_to_u(
        w: Float[Array, "num_points num_seeds"],
    ) -> Float[Array, "num_points num_seeds_minus_one"]:
        return weight_chart_inverse(w, tolerance=tol)  # type: ignore[call-arg]

    @jaxtyped(typechecker=beartype)
    def from_w_to_z(
        w: Float[Array, "num_points num_seeds"],
    ) -> Float[Array, "num_points num_latent_dims"]:
        inner_latent = from_w_to_inner_latent(
            w=w,
            inner_seed_latents=inner_seed_latents,
            tolerance=tol,
        )
        return np.stack([transport_map_backward(p) for p in inner_latent], axis=0)

    @jaxtyped(typechecker=beartype)
    def from_z_to_w(
        z: Float[Array, "num_points num_latent_dims"],
    ) -> Float[Array, "num_points num_seeds"]:
        inner_latent = np.stack([transport_map_forward(p) for p in z], axis=0)
        w = from_inner_latent_to_w(
            inner_latent=inner_latent,
            inner_seed_latents_pseudoinverse=inner_seed_latents_pinv,
            tolerance=tol,
        )
        return adjust_weights_to_positive_orthant(w, tolerance=tol)

    @jaxtyped(typechecker=beartype)
    def from_u_to_z(
        u: Float[Array, "num_points num_seeds_minus_one"],
    ) -> Float[Array, "num_points num_latent_dims"]:
        return from_w_to_z(from_u_to_w(u))

    @jaxtyped(typechecker=beartype)
    def from_z_to_u(
        z: Float[Array, "num_points num_latent_dims"],
    ) -> Float[Array, "num_points num_seeds_minus_one"]:
        return from_w_to_u(from_z_to_w(z))

    return {
        "num_seeds": num_seeds,
        "num_latent_dims": num_latent_dims,
        "from_u_to_z": from_u_to_z,
        "from_z_to_u": from_z_to_u,
        "from_u_to_w": from_u_to_w,
        "from_w_to_u": from_w_to_u,
        "from_w_to_z": from_w_to_z,
        "from_z_to_w": from_z_to_w,
    }


def simplex_surrogate_chart(
    seed_latents: list[Any],
    *,
    transport_map_forward: Callable[[Any], Float[Array, "num_latent_dims"]] = lambda x: x,
    transport_map_backward: Callable[[Float[Array, "num_latent_dims"]], Any] = lambda x: x,
) -> dict[str, Any]:
    return make_surrogate_chart(
        seed_latents=seed_latents,
        transport_map_forward=transport_map_forward,
        transport_map_backward=transport_map_backward,
        weight_chart_forward=simplex_chart_forward,
        weight_chart_inverse=simplex_chart_inverse,
    )


def angular_surrogate_chart(
    seed_latents: list[Any],
    *,
    transport_map_forward: Callable[[Any], Float[Array, "num_latent_dims"]] = lambda x: x,
    transport_map_backward: Callable[[Float[Array, "num_latent_dims"]], Any] = lambda x: x,
) -> dict[str, Any]:
    return make_surrogate_chart(
        seed_latents=seed_latents,
        transport_map_forward=transport_map_forward,
        transport_map_backward=transport_map_backward,
        weight_chart_forward=angular_chart_forward,
        weight_chart_inverse=angular_chart_inverse,
    )


def knothe_rosenblatt_surrogate_chart(
    seed_latents: list[Any],
    *,
    transport_map_forward: Callable[[Any], Float[Array, "num_latent_dims"]] = lambda x: x,
    transport_map_backward: Callable[[Float[Array, "num_latent_dims"]], Any] = lambda x: x,
) -> dict[str, Any]:
    return make_surrogate_chart(
        seed_latents=seed_latents,
        transport_map_forward=transport_map_forward,
        transport_map_backward=transport_map_backward,
        weight_chart_forward=knothe_rosenblatt_forward,
        weight_chart_inverse=knothe_rosenblatt_inverse,
    )

