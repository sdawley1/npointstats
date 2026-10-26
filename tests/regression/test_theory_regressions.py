#!/usr/bin/env python3

"""
Regression tests for the liquid-state theory defects D-034 … D-040.

Each test states the assertion that failed against the original code in
snapshot 363ae8f (utils/approx/*.py, utils/nft/numerical_fourier_transform.py).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import importlib

# ----- third party -----
import numpy as np
import pytest

# ----- local modules -----
from npointstats.theory import baus_colot as bc
from npointstats.theory import (
    bc_dcf,
    bc_dcf_fourier,
    bc_inverse_compressibility,
    bc_solve_a,
    bc_structure_factor,
    inverse_radial_fourier_transform,
    number_density,
    py_dcf,
    py_dcf_fourier,
    py_structure_factor,
    radial_fourier_transform,
)

# ============================================================
# HELPERS
# ============================================================


def _complex_step_k(eta: float, d: int) -> float:
    """Exact d(ηZ)/dη by complex step from the stored EOS numerators."""
    h = 1e-30
    z = eta + 1j * h
    p = np.polynomial.polynomial.polyval(z, bc.EOS_NUMERATORS[d])
    return float((z * p / (1.0 - z) ** d).imag / h)


# ============================================================
# D-034: number density from packing fraction
# ============================================================


def test_regression_D034_number_density_no_root() -> None:
    """D-034: ρ = 4η/(πD²) in 2D and 6η/(πD³) in 3D, not their square/cube roots."""
    np.testing.assert_allclose(number_density(0.5, 2, 1.0), 2.0 / np.pi, rtol=1e-12)
    np.testing.assert_allclose(number_density(0.5, 3, 1.0), 3.0 / np.pi, rtol=1e-12)
    np.testing.assert_allclose(number_density(0.5, 1, 2.0), 0.25, rtol=1e-12)


# ============================================================
# D-035: Jacobian of the a-equation
# ============================================================


def test_regression_D035_a_equation_jacobian() -> None:
    """D-035: fprime of the a-residual equals its derivative (η = 0.4, a = 1.5)."""
    eta, a, h = 0.4, 1.5, 1e-6
    k_inv = bc_inverse_compressibility(eta, 2)
    fd = bc._bc_residual_2d(a + h, eta, k_inv) - bc._bc_residual_2d(a - h, eta, k_inv)
    np.testing.assert_allclose(bc._bc_jacobian_2d(a, eta), fd / (2 * h), rtol=1e-6)
    # d = 3: the residual 9η²a² - 2η² - 8η + 1 - 1/K is solved in closed form
    k3 = bc_inverse_compressibility(eta, 3)
    a3 = bc_solve_a(eta, 3)
    residual = 9 * (eta * a3) ** 2 - 2 * eta**2 - 8 * eta + 1 - 1 / k3
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
    # the 2D root is a true root of the residual
    a2 = bc_solve_a(eta, 2)
    np.testing.assert_allclose(bc._bc_residual_2d(a2, eta, k_inv), 0.0, atol=1e-14)


# ============================================================
# D-036: analytic d(ηZ)/dη
# ============================================================


@pytest.mark.parametrize(("eta", "d"), [(0.45, 3), (0.85, 2)])
def test_regression_D036_analytic_compressibility_derivative(
    eta: float, d: int
) -> None:
    """D-036: K = d(ηZ)/dη is exact at η itself (was finite-differenced off-grid)."""
    expected = _complex_step_k(eta, d)
    np.testing.assert_allclose(bc_inverse_compressibility(eta, d), expected, rtol=1e-10)


# ============================================================
# D-037: Lado sums include node N - 1
# ============================================================


@pytest.mark.parametrize("d", [1, 3])
def test_regression_D037_lado_sums_keep_last_node(
    d: int, rng: np.random.Generator
) -> None:
    """D-037: forward∘inverse is the identity on n = N - 1 nodes, n as requested."""
    n = 40
    x = rng.normal(size=n)
    k, F = radial_fourier_transform(x, d, 10.0, n)
    r, y = inverse_radial_fourier_transform(F, d, 10.0, n)
    np.testing.assert_allclose(y, x, rtol=0.0, atol=1e-10)
    assert k.shape == (n,)
    assert r.shape == (n,)


# ============================================================
# D-038: Percus–Yevick importable; d = 2 raises
# ============================================================


def test_regression_D038_percus_yevick_importable_and_2d_raises() -> None:
    """D-038: module imports, d = 1/3 evaluate, d = 2 raises NotImplementedError."""
    module = importlib.import_module("npointstats.theory.percus_yevick")
    assert callable(module.py_dcf)
    r = np.linspace(0.0, 2.0, 5)
    assert np.all(np.isfinite(py_dcf(r, 0.3, 1)))
    assert np.all(np.isfinite(py_dcf_fourier(r, 0.3, 3)))
    with pytest.raises(NotImplementedError, match="hard disks"):
        py_dcf(r, 0.3, 2)


# ============================================================
# D-039: one ρ convention for c̃(k)
# ============================================================


def test_regression_D039_dcf_fourier_excludes_density() -> None:
    """D-039: BC and PY both return c̃(k) without ρ; equal to exact Tonks in 1D."""
    eta = 0.4
    q = np.linspace(0.5, 20.0, 9)
    f, g = np.sin(q) / q, np.sin(q / 2) / (q / 2)
    tonks = -(2 * (1 - eta) * f + eta * g**2) / (1 - eta) ** 2
    np.testing.assert_allclose(bc_dcf_fourier(q, eta, 1), tonks, rtol=1e-12)
    np.testing.assert_allclose(py_dcf_fourier(q, eta, 1), tonks, rtol=1e-12)
    np.testing.assert_allclose(
        bc_structure_factor(q, eta, 1), py_structure_factor(q, eta, 1), rtol=1e-12
    )


# ============================================================
# D-040: concrete dispatchers reject unsupported d
# ============================================================


@pytest.mark.parametrize("func", [bc_dcf, bc_dcf_fourier, py_dcf, py_dcf_fourier])
def test_regression_D040_unsupported_dimension_raises(func: object) -> None:
    """D-040: d = 4 raises ValueError instead of returning 1."""
    with pytest.raises(ValueError, match="dimension d must be 1, 2, or 3"):
        func(0.5, 0.3, 4)  # type: ignore[operator]
