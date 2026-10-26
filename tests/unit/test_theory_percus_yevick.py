#!/usr/bin/env python3

"""
Unit tests for the Percus–Yevick hard-rod and hard-sphere closures.

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
from npointstats.theory import (
    lado_grid,
    number_density,
    py_dcf,
    py_dcf_fourier,
    py_structure_factor,
    radial_fourier_transform,
)

# ============================================================
# HELPERS
# ============================================================


def _quad_transform(eta: float, q: float, d: int) -> float:
    """c̃(q) of the PY DCF by adaptive quadrature over [0, 1] (D = 1)."""

    def integrand(r: float) -> float:
        value = float(py_dcf(r, eta, d))
        if d == 1:
            return 2.0 * value * np.cos(q * r)
        return 4.0 * np.pi * r * r * value * float(spherical_jn(0, q * r))

    return quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13, limit=400)[0]


# ============================================================
# d = 3
# ============================================================


@pytest.mark.parametrize("eta", [0.0, 0.05, 0.2, 0.35, 0.45, 0.5])
def test_s0_compressibility_route_3d(eta: float) -> None:
    expected = (1 - eta) ** 4 / (1 + 2 * eta) ** 2
    np.testing.assert_allclose(py_structure_factor(0.0, eta, 3), expected, rtol=1e-13)


def test_wertheim_contact_values_3d() -> None:
    eta = 0.4
    lam1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    lam2 = (1 + eta / 2) ** 2 / (1 - eta) ** 4
    np.testing.assert_allclose(py_dcf(0.0, eta, 3), -lam1, rtol=1e-15)
    inner = -lam1 + 6 * eta * lam2 - 0.5 * eta * lam1
    np.testing.assert_allclose(py_dcf(1.0 - 1e-14, eta, 3), inner, rtol=1e-12)


@pytest.mark.parametrize("d", [1, 3])
def test_dcf_zero_outside_core(d: int) -> None:
    r = np.array([1.5 + 1e-12, 2.0, 7.0])
    np.testing.assert_array_equal(py_dcf(r, 0.3, d, diameter=1.5), 0.0)
    assert np.all(py_dcf(np.array([0.0, 1.4]), 0.3, d, diameter=1.5) < 0.0)


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("eta", [0.1, 0.45])
def test_fourier_matches_quadrature(d: int, eta: float) -> None:
    for q in (0.0, 1e-4, 1.0, 2.0, 5.0, 30.0):
        expected = _quad_transform(eta, q, d)
        got = float(py_dcf_fourier(q, eta, d))
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_fourier_matches_lado_transform_3d() -> None:
    # Δr = 0.01 puts r = D on a node, where c takes its midpoint value; the
    # Lado sum is then a trapezoid rule with O(Δr²) error.
    eta = 0.3
    k, F = radial_fourier_transform(lambda r: py_dcf(r, eta, 3), 3, 10.0, 999)
    exact = py_dcf_fourier(k, eta, 3)
    scale = np.max(np.abs(exact))
    np.testing.assert_allclose(F[:200], exact[:200], rtol=0.0, atol=1e-3 * scale)


# ============================================================
# d = 1
# ============================================================


@pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9])
def test_1d_equals_tonks(eta: float) -> None:
    np.testing.assert_allclose(py_structure_factor(0.0, eta, 1), (1 - eta) ** 2)
    q = np.array([0.5, 3.0, 11.0])
    f = np.sin(q) / q
    g = np.sin(q / 2) / (q / 2)
    tonks = -(2 * (1 - eta) * f + eta * g**2) / (1 - eta) ** 2
    np.testing.assert_allclose(py_dcf_fourier(q, eta, 1), tonks, rtol=1e-12)


def test_1d_lado_grid_transform_agrees() -> None:
    # R = Δr (n + 1/2) with Δr = 0.005 puts r = D on a cell edge of the
    # midpoint rule, so the jump of c costs only O(Δr²).
    n = 2000
    r_max = 0.005 * (n + 0.5)
    r, _ = lado_grid(1, r_max, n)
    k, F = radial_fourier_transform(py_dcf(r, 0.5, 1), 1, r_max, n)
    exact = py_dcf_fourier(k, 0.5, 1)
    np.testing.assert_allclose(F[:50], exact[:50], rtol=0.0, atol=1e-4)


# ============================================================
# SHARED BEHAVIOUR
# ============================================================


@pytest.mark.parametrize("d", [1, 3])
def test_structure_factor_tends_to_one(d: int) -> None:
    s = py_structure_factor(np.array([1e3, 1e4, 1e5]), 0.4, d)
    np.testing.assert_allclose(s, 1.0, atol=1e-2)
    assert abs(s[-1] - 1.0) < 1e-4


@pytest.mark.parametrize("d", [1, 3])
def test_density_factor_not_included(d: int) -> None:
    # c̃ carries units of D^d and no ρ: at η = 0, c = -θ(D - r), c̃(0) = -v_1(D).
    volume = {1: 2.0, 3: 4.0 * np.pi / 3.0}[d] * 2.0**d
    np.testing.assert_allclose(py_dcf_fourier(0.0, 0.0, d, 2.0), -volume)
    rho = number_density(0.3, d, 2.0)
    s = 1.0 / (1.0 - rho * py_dcf_fourier(0.0, 0.3, d, 2.0))
    np.testing.assert_allclose(py_structure_factor(0.0, 0.3, d, 2.0), s)


# ============================================================
# ERROR CONTRACT
# ============================================================


@pytest.mark.parametrize("func", [py_dcf, py_dcf_fourier, py_structure_factor])
def test_2d_raises_not_implemented(func: object) -> None:
    with pytest.raises(NotImplementedError, match="baus_colot"):
        func(0.5, 0.3, 2)  # type: ignore[operator]


@pytest.mark.parametrize("func", [py_dcf, py_dcf_fourier, py_structure_factor])
@pytest.mark.parametrize("d", [0, 4, -1])
def test_unsupported_dimension_raises_value_error(func: object, d: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        func(0.5, 0.3, d)  # type: ignore[operator]


@pytest.mark.parametrize("kwargs", [{"eta": 1.0}, {"eta": -1e-3}, {"eta": True},
                                    {"diameter": 0.0}, {"r": -1.0}])  # fmt: skip
def test_invalid_state_raises_value_error(kwargs: dict) -> None:
    args = {"r": 0.5, "eta": 0.3, "d": 3, "diameter": 1.0} | kwargs
    with pytest.raises(ValueError, match=r"eta|diameter|r must|packing"):
        py_dcf(**args)
