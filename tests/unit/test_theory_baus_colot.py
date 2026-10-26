#!/usr/bin/env python3

"""
Unit tests for the Baus–Colot equation of state, DCF, and structure factor.

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
from scipy.special import j0, spherical_jn

# ----- local modules -----
from npointstats.theory import baus_colot as bc
from npointstats.theory import (
    bc_compressibility_factor,
    bc_dcf,
    bc_dcf_fourier,
    bc_inverse_compressibility,
    bc_solve_a,
    bc_structure_factor,
    number_density,
    py_dcf_fourier,
)

# ============================================================
# CONSTANTS
# ============================================================

# Baus & Colot (1987), Tables IV (d = 2) and V (d = 3): (η_cp / η, Z_6).
TABLE_2D = [(30, 1.06344), (20, 1.09754), (10, 1.21068), (5, 1.49842),
            (3, 2.07721), (2, 3.42452), (1.8, 4.17193), (1.6, 5.49813),
            (1.5, 6.61433), (1.4, 8.35233)]  # fmt: skip
TABLE_3D = [(25, 1.127750), (18, 1.182839), (10, 1.359427), (5, 1.888516),
            (4, 2.244506), (3, 3.033017), (2, 5.877755), (1.8, 7.495420),
            (1.7, 8.706121), (1.6, 10.373926)]  # fmt: skip
ETA_CP = {2: np.pi / (2.0 * np.sqrt(3.0)), 3: np.pi / (3.0 * np.sqrt(2.0))}

# ============================================================
# HELPERS
# ============================================================


def _quad_transform(c: object, q: float, d: int, diameter: float = 1.0) -> float:
    """c̃(q) = ∫ c(r) e^{-iq·r} d^d r by adaptive quadrature over [0, D]."""

    def integrand(r: float) -> float:
        value = float(c(r))  # type: ignore[operator]
        if d == 1:
            return 2.0 * value * np.cos(q * r)
        if d == 2:
            return 2.0 * np.pi * r * value * float(j0(q * r))
        return 4.0 * np.pi * r * r * value * float(spherical_jn(0, q * r))

    return quad(integrand, 0.0, diameter, epsabs=1e-13, epsrel=1e-13, limit=400)[0]


def _complex_step_k(eta: float, d: int) -> float:
    """d(ηZ)/dη by complex step (exact to round-off) from the stored numerators."""
    h = 1e-30
    z = eta + 1j * h
    p = np.polynomial.polynomial.polyval(z, bc.EOS_NUMERATORS[d])
    return float((z * p / (1.0 - z) ** d).imag / h)


# ============================================================
# EQUATION OF STATE
# ============================================================


def test_number_density_closed_forms() -> None:
    np.testing.assert_allclose(number_density(0.3, 1, 2.0), 0.15, rtol=1e-15)
    np.testing.assert_allclose(number_density(0.3, 2), 1.2 / np.pi, rtol=1e-15)
    np.testing.assert_allclose(number_density(0.3, 3, 0.5), 14.4 / np.pi, rtol=1e-14)


@pytest.mark.parametrize("d", [2, 3])
def test_compressibility_factor_reproduces_baus_colot_tables(d: int) -> None:
    table = TABLE_2D if d == 2 else TABLE_3D
    for ratio, z_paper in table:
        z = bc_compressibility_factor(ETA_CP[d] / ratio, d)
        np.testing.assert_allclose(z, z_paper, rtol=2e-5)  # 4-digit coefficients


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9])
def test_compressibility_factor_is_tonks_in_1d(eta: float) -> None:
    np.testing.assert_allclose(bc_compressibility_factor(eta, 1), 1 / (1 - eta))


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.0, 0.05, 0.3, 0.5, 0.7])
def test_inverse_compressibility_is_exact_derivative(d: int, eta: float) -> None:
    expected = _complex_step_k(eta, d)
    np.testing.assert_allclose(bc_inverse_compressibility(eta, d), expected, rtol=1e-13)


# ============================================================
# PARAMETER a
# ============================================================


@pytest.mark.parametrize("d", [2, 3])
def test_a_tends_to_two_at_low_density(d: int) -> None:
    np.testing.assert_allclose(bc_solve_a(1e-4, d), 2.0, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("eta", [0.01, 0.2, 0.45, 0.7, 0.9])
def test_a_lies_between_one_and_two(d: int, eta: float) -> None:
    assert 1.0 < bc_solve_a(eta, d) < 2.0


def test_solve_a_rejects_1d_and_zero_density() -> None:
    with pytest.raises(ValueError, match="undefined in d = 1"):
        bc_solve_a(0.3, 1)
    with pytest.raises(ValueError, match="0 < eta < 1"):
        bc_solve_a(0.0, 3)


# ============================================================
# DCF AND S(k)
# ============================================================


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.05, 0.3, 0.5])
def test_compressibility_route_is_reproduced(d: int, eta: float) -> None:
    rho = number_density(eta, d)
    k_inv = bc_inverse_compressibility(eta, d)
    analytic = 1.0 - rho * float(bc_dcf_fourier(0.0, eta, d))
    numeric = 1.0 - rho * _quad_transform(lambda r: bc_dcf(r, eta, d), 0.0, d)
    np.testing.assert_allclose(analytic, k_inv, rtol=1e-12)
    np.testing.assert_allclose(numeric, k_inv, rtol=1e-10)
    np.testing.assert_allclose(float(bc_structure_factor(0.0, eta, d)), 1 / k_inv)


@pytest.mark.parametrize("eta", [0.1, 0.4, 0.8])
def test_1d_is_exact_tonks(eta: float) -> None:
    r = np.linspace(0.0, 0.99, 7)
    np.testing.assert_allclose(bc_dcf(r, eta, 1), -(1 - eta * r) / (1 - eta) ** 2)
    k = np.array([0.0, 0.5, 3.0, 17.0])
    np.testing.assert_allclose(
        bc_dcf_fourier(k, eta, 1), py_dcf_fourier(k, eta, 1), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(bc_structure_factor(0.0, eta, 1), (1 - eta) ** 2)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("eta", [0.05, 0.45])
def test_fourier_matches_quadrature_of_dcf(d: int, eta: float) -> None:
    for q in (0.3, 1.9, 2.1, 6.5, 25.0):
        expected = _quad_transform(lambda r: bc_dcf(r, eta, d), q, d)
        got = float(bc_dcf_fourier(q, eta, d))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_dcf_vanishes_outside_core_and_halves_at_contact(d: int) -> None:
    r = np.array([0.0, 1.0 - 1e-13, 2.0, 2.0 + 1e-9, 3.0, 50.0])
    c = bc_dcf(r, 0.4, d, diameter=2.0)
    np.testing.assert_array_equal(c[3:], 0.0)
    np.testing.assert_allclose(c[2], 0.5 * bc_dcf(2.0 - 1e-12, 0.4, d, 2.0), rtol=1e-9)
    assert np.all(c[:2] < 0.0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_structure_factor_tends_to_one(d: int) -> None:
    s = bc_structure_factor(np.array([1e3, 1e4]), 0.45, d)
    np.testing.assert_allclose(s, 1.0, atol=0.02)
    assert abs(s[1] - 1.0) < 2e-3


def test_diameter_scaling() -> None:
    k = np.array([0.0, 0.7, 4.0])
    for d in (1, 2, 3):
        scaled = bc_dcf_fourier(k / 1.7, 0.35, d, diameter=1.7)
        np.testing.assert_allclose(scaled, 1.7**d * bc_dcf_fourier(k, 0.35, d))


# ============================================================
# ERROR CONTRACT
# ============================================================


@pytest.mark.parametrize("func", [bc_dcf, bc_dcf_fourier, bc_structure_factor])
@pytest.mark.parametrize("d", [0, 4, 5])
def test_unsupported_dimension_raises_value_error(func: object, d: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        func(0.5, 0.3, d)  # type: ignore[operator]


@pytest.mark.parametrize("kwargs", [{"eta": 0.0}, {"eta": 1.0}, {"eta": -0.1},
                                    {"eta": np.nan}, {"diameter": 0.0},
                                    {"diameter": -1.0}, {"r": -0.5},
                                    {"r": np.nan}])  # fmt: skip
def test_invalid_state_raises_value_error(kwargs: dict) -> None:
    args = {"r": 0.5, "eta": 0.3, "d": 3, "diameter": 1.0} | kwargs
    with pytest.raises(ValueError, match=r"eta|diameter|r must"):
        bc_dcf(**args)
