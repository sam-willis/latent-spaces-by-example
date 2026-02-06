import numpy as np

from latent_spaces_by_example import knothe_rosenblatt_surrogate_chart


def main() -> None:
    rng = np.random.default_rng(0)

    n_seeds = 5
    latent_dim = 3
    seed_latents = rng.normal(size=(n_seeds, latent_dim))

    chart = knothe_rosenblatt_surrogate_chart(seed_latents)

    n_points = 10
    u = rng.uniform(size=(n_points, n_seeds - 1))
    z = chart.from_u_to_z(u)

    print("u shape:", u.shape)
    print("z shape:", z.shape)
    print("first z:", z[0])

    # Round-trip sanity (not guaranteed exact; should be close for KR chart)
    u2 = chart.from_z_to_u(z)
    max_abs_err = np.max(np.abs(u2 - u))
    print("max |u2 - u|:", max_abs_err)


if __name__ == "__main__":
    main()
