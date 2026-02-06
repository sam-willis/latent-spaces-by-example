import numpy as np

from latent_spaces_by_example.transport_maps import IdentityTransportMap


def test_identity_transport_map_forward_backward_identity() -> None:
    tm = IdentityTransportMap()
    z = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)

    xi = tm.forward(z)
    z2 = tm.backward(xi)

    assert xi.shape == z.shape
    assert z2.shape == z.shape
    assert np.allclose(xi, z)
    assert np.allclose(z2, z)
