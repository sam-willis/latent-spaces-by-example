import numpy as np

from latent_spaces_by_example.utils import (
    adjust_weights_to_positive_orthant,
    is_on_hypersphere,
)


def test_adjust_weights_to_positive_orthant_clips_and_normalizes() -> None:
    w = np.array(
        [
            [-1.0, 2.0, 0.0],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    w2 = adjust_weights_to_positive_orthant(w, tolerance=1e-12)

    assert w2.shape == w.shape
    assert w2.dtype == np.float64
    assert np.all(w2 >= 0.0)
    assert np.allclose(np.linalg.norm(w2, axis=-1), 1.0, atol=1e-12)


def test_adjust_weights_to_positive_orthant_all_zero_row_is_finite() -> None:
    w = np.array([[-1.0, -2.0, -3.0]], dtype=np.float64)
    w2 = adjust_weights_to_positive_orthant(w, tolerance=1e-12)

    assert np.all(np.isfinite(w2))
    assert np.all(w2 == 0.0)
    assert np.allclose(np.linalg.norm(w2, axis=-1), 0.0)


def test_is_on_hypersphere_true_and_false_cases() -> None:
    x = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    assert is_on_hypersphere(x, tolerance=1e-12) is True

    y = np.array([[1.0, 0.0], [0.0, 0.9]], dtype=np.float64)
    assert is_on_hypersphere(y, tolerance=1e-12) is False


def test_is_on_hypersphere_tolerance_boundary() -> None:
    # Avoid equality-to-tolerance brittleness from floating point rounding.
    tol = 1e-6
    x = np.array([[1.0 + 2.0 * tol, 0.0]], dtype=np.float64)
    assert is_on_hypersphere(x, tolerance=tol) is False

    y = np.array([[1.0 + 0.25 * tol, 0.0]], dtype=np.float64)
    assert is_on_hypersphere(y, tolerance=tol) is True
