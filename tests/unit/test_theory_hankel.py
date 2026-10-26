#!/usr/bin/env python3

"""
Unit tests for the Lado radial Fourier transforms and the ball transform.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- third party -----
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

# ----- local modules -----
from npointstats.theory.hankel import (
    ball_polynomial_transform,
    inverse_radial_fourier_transform,
    lado_grid,
    radial_fourier_transform,
)

# ============================================================
# HELPERS
# ============================================================


def _gaussian(r: np.ndarray) -> np.ndarray:
    """f(r) = exp(-r²/2)."""
    return np.exp(-(r**2) / 2.0)


def _gaussian_hat(k: np.ndarray, d: int) -> np.ndarray:
    """Analytic transform (2π)^{d/2} exp(-k²/2) of `_gaussian`."""
    return (2.0 * np.pi) ** (d / 2.0) * np.exp(-(k**2) / 2.0)


# ============================================================
# GRIDS
# ============================================================


@pytest.mark.parametrize("d", [1, 2, 3])
def test_lado_grid_has_exactly_n_increasing_nodes(d: int) -> None:
    r, k = lado_grid(d, 7.5, 37)
    assert r.shape == (37,)
    assert k.shape == (37,)
    assert np.all(np.diff(r) > 0.0)
    assert np.all(np.diff(k) > 0.0)
    assert r[0] > 0.0
    assert r[-1] < 7.5


def test_lado_grid_spacings_d1_d3() -> None:
    r1, k1 = lado_grid(1, 10.0, 9)
    np.testing.assert_allclose(r1[0], 0.5 * 10.0 / 9.5, rtol=1e-15)
    np.testing.assert_allclose(np.diff(k1), np.pi / 10.0, rtol=1e-13)
    r3, k3 = lado_grid(3, 10.0, 9)
    np.testing.assert_allclose(r3, np.arange(1, 10), rtol=1e-15)
    np.testing.assert_allclose(k3, np.arange(1, 10) * np.pi / 10.0, rtol=1e-15)


# ============================================================
# ROUND TRIP AND GAUSSIAN
# ============================================================


@pytest.mark.parametrize("d", [1, 3])
def test_round_trip_is_exact_for_any_samples(d: int, rng: np.random.Generator) -> None:
    x = rng.normal(size=150)
    _, F = radial_fourier_transform(x, d, 10.0, 150)
    _, y = inverse_radial_fourier_transform(F, d, 10.0, 150)
    np.testing.assert_allclose(y, x, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_round_trip_gaussian_on_grid(d: int) -> None:
    r, _ = lado_grid(d, 10.0, 200)
    f = _gaussian(r)
    _, F = radial_fourier_transform(f, d, 10.0, 200)
    _, g = inverse_radial_fourier_transform(F, d, 10.0, 200)
    np.testing.assert_allclose(g, f, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_forward_gaussian_matches_analytic(d: int) -> None:
    k, F = radial_fourier_transform(_gaussian, d, 10.0, 200)
    np.testing.assert_allclose(F, _gaussian_hat(k, d), rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_inverse_gaussian_matches_analytic(d: int) -> None:
    r, f = inverse_radial_fourier_transform(lambda k: _gaussian_hat(k, d), d, 10.0, 200)
    np.testing.assert_allclose(f, _gaussian(r), rtol=0.0, atol=1e-10)


def test_callable_and_samples_give_same_transform() -> None:
    r, _ = lado_grid(2, 8.0, 50)
    k1, F1 = radial_fourier_transform(_gaussian, 2, 8.0, 50)
    k2, F2 = radial_fourier_transform(_gaussian(r), 2, 8.0, 50)
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_allclose(F1, F2, rtol=1e-15, atol=0.0)


def test_input_samples_are_not_mutated() -> None:
    x = np.linspace(1.0, 0.0, 20)
    snapshot = x.copy()
    radial_fourier_transform(x, 3, 5.0, 20)
    np.testing.assert_array_equal(x, snapshot)


# ============================================================
# ERROR CONTRACT
# ============================================================


@pytest.mark.parametrize(
    ("d", "r_max", "n"),
    [(0, 1.0, 10), (4, 1.0, 10), (True, 1.0, 10), (1.0, 1.0, 10), (1, 0.0, 10),
     (1, -2.0, 10), (1, np.nan, 10), (1, np.inf, 10), (1, 1.0, 1), (1, 1.0, 2.5)],
)  # fmt: skip
def test_invalid_grid_arguments_raise(d: int, r_max: float, n: int) -> None:
    with pytest.raises(ValueError, match=r"dimension|r_max|n must"):
        radial_fourier_transform(np.zeros(10), d, r_max, n)


def test_wrong_sample_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        radial_fourier_transform(np.zeros(9), 1, 1.0, 10)


def test_non_finite_samples_raise() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        inverse_radial_fourier_transform(np.full(10, np.nan), 3, 1.0, 10)


# ============================================================
# BALL POLYNOMIAL TRANSFORM
# ============================================================


def test_ball_transform_at_zero_is_ball_volume() -> None:
    np.testing.assert_allclose(ball_polynomial_transform([1.0], 0.0, 1), 2.0)
    np.testing.assert_allclose(
        ball_polynomial_transform([1.0], 0.0, 3), 4.0 * np.pi / 3.0, rtol=1e-15
    )


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("k", [0.0, 1e-9, 1e-3, 0.7, 1.999, 2.0, 2.001, 6.0, 35.0])
def test_ball_transform_matches_quadrature(d: int, k: float) -> None:
    coeffs = np.array([0.3, -1.2, 0.0, 0.7, 0.25])

    def integrand(x: float) -> float:
        p = float(np.polynomial.polynomial.polyval(x, coeffs))
        if d == 1:
            return 2.0 * p * np.cos(k * x)
        return 4.0 * np.pi * p * x * x * float(spherical_jn(0, k * x))

    expected = quad(integrand, 0.0, 1.0, epsabs=1e-14, epsrel=1e-14, limit=200)[0]
    got = float(ball_polynomial_transform(coeffs, k, d))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


def test_ball_transform_keeps_input_shape() -> None:
    k = np.linspace(0.0, 10.0, 12).reshape(3, 4)
    assert ball_polynomial_transform([1.0, 2.0], k, 3).shape == (3, 4)


@pytest.mark.parametrize("bad", [{"d": 2}, {"coeffs": []}, {"coeffs": [[1.0]]},
                                 {"coeffs": [np.nan]}, {"k": np.inf}])  # fmt: skip
def test_ball_transform_invalid_inputs_raise(bad: dict) -> None:
    args = {"coeffs": [1.0], "k": 1.0, "d": 3} | bad
    with pytest.raises(ValueError, match=r"d = 1 or 3|coeffs|k must"):
        ball_polynomial_transform(args["coeffs"], args["k"], args["d"])
