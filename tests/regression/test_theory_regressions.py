#!/usr/bin/env python3

"""
Regression tests for the liquid-state theory defects D-034 … D-040, D-042 … D-046.

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
from scipy.integrate import quad
from scipy.special import j0, spherical_jn

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


# ============================================================
# D-042: 2D Baus–Colot c̃(k) available under NumPy 2
# ============================================================


def test_regression_D042_bc_2d_fourier_numpy2() -> None:
    """D-042: 2D c̃ evaluates (no np.trapz) and equals the transform of c(r)."""
    eta = 0.4
    q = np.linspace(0.5, 5.0, 5)
    got = bc_dcf_fourier(q, eta, 2)
    expected = [_quad_transform(lambda r: bc_dcf(r, eta, 2), x, 2) for x in q]
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-11)
    # edges: k = 0 obeys the sum rule; large k stays finite and S -> 1
    rho = number_density(eta, 2)
    k_inv = bc_inverse_compressibility(eta, 2)
    np.testing.assert_allclose(1 - rho * bc_dcf_fourier(0.0, eta, 2), k_inv, rtol=1e-12)
    np.testing.assert_allclose(bc_structure_factor(500.0, eta, 2), 1.0, atol=1e-3)


# ============================================================
# D-043: non-converged a must never be used
# ============================================================


@pytest.mark.parametrize(
    ("eta", "a_ref"),
    [(0.8, 1.1145762880898777), (0.85, 1.0830881913146504), (0.9, 1.0535595980177)],
)
def test_regression_D043_bc_2d_a_converges_at_high_density(
    eta: float, a_ref: float
) -> None:
    """D-043: 2D a at η >= 0.8 is the true root (fsolve's guess 1.5 was used)."""
    a = bc_solve_a(eta, 2)
    np.testing.assert_allclose(a, a_ref, rtol=1e-10)
    k_inv = bc_inverse_compressibility(eta, 2)
    np.testing.assert_allclose(bc._bc_residual_2d(a, eta, k_inv), 0.0, atol=1e-12)
    # independent check: the resulting c(r) satisfies 1 - ρ ∫ c = K
    integral = _quad_transform(lambda r: bc_dcf(r, eta, 2), 0.0, 2)
    np.testing.assert_allclose(1 - number_density(eta, 2) * integral, k_inv, rtol=1e-9)


# ============================================================
# D-044: evaluate exactly at the caller's r and k
# ============================================================


def test_regression_D044_outputs_at_caller_grid() -> None:
    """D-044: scalar r gives a scalar at that r; array k is used as given."""
    eta = 0.3
    c = bc_dcf(0.5, eta, 3)
    assert c.shape == ()
    k_inv, a = bc_inverse_compressibility(eta, 3), bc_solve_a(eta, 3)
    expected = -k_inv * (1 - 1.5 * eta * a**2 * 0.5 + 0.5 * eta * 0.125)
    np.testing.assert_allclose(float(c), expected, rtol=1e-14)
    q = np.array([1.0, 2.0, 3.0])
    ck = bc_dcf_fourier(q, eta, 3)
    assert ck.shape == (3,)
    for x, value in zip(q, ck, strict=True):
        np.testing.assert_allclose(value, float(bc_dcf_fourier(float(x), eta, 3)))
        expected_q = _quad_transform(lambda r: bc_dcf(r, eta, 3), x, 3)
        np.testing.assert_allclose(value, expected_q, rtol=1e-10)
    # edge: an array r with no k given keeps its own values and shape
    r = np.array([[0.2, 0.4], [1.5, 0.0]])
    out = bc_dcf(r, eta, 3)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0, 1], float(bc_dcf(0.4, eta, 3)))
    assert py_dcf(0.5, eta, 3).shape == ()


# ============================================================
# D-045: c̃(k) at and near k = 0
# ============================================================


def test_regression_D045_small_k_no_cancellation() -> None:
    """D-045: PY c̃(k) at k = 0, 1e-5, 1e-3 equals its Taylor series to 1e-9."""
    eta, lam1, lam2 = 0.4, 25.0, 100.0 / 9.0  # λ1, λ2 at η = 0.4
    second = -lam1 / 5 + eta * lam2 - eta * lam1 / 16  # ∫_0^1 x^4 c(x) dx
    for k in (1e-3, 1e-5):
        expected = -10 * np.pi - (4 * np.pi / 6) * k * k * second
        np.testing.assert_allclose(
            float(py_dcf_fourier(k, eta, 3)), expected, rtol=1e-9
        )
    np.testing.assert_allclose(
        float(py_dcf_fourier(0.0, eta, 3)), -10 * np.pi, rtol=1e-14
    )
    # edges: every closure is finite at k = 0 and continuous across k = 2
    for func, d in ((py_dcf_fourier, 1), (bc_dcf_fourier, 1), (bc_dcf_fourier, 2),
                    (bc_dcf_fourier, 3)):  # fmt: skip
        assert np.isfinite(func(0.0, eta, d))
        np.testing.assert_allclose(
            func(2.0 - 1e-12, eta, d), func(2.0 + 1e-12, eta, d), rtol=1e-10
        )


# ============================================================
# D-046: the diameter enters c(r) and c̃(k)
# ============================================================


@pytest.mark.parametrize("d", [1, 2, 3])
def test_regression_D046_diameter_is_used(d: int) -> None:
    """D-046: c(1.5; D = 2) is inside the core and c̃(k; D) = D^d c̃(kD; 1)."""
    eta = 0.3
    c = float(bc_dcf(1.5, eta, d, diameter=2.0))
    assert c < 0.0
    np.testing.assert_allclose(c, float(bc_dcf(0.75, eta, d)), rtol=1e-14)
    q = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        bc_dcf_fourier(q, eta, d, diameter=2.0), 2.0**d * bc_dcf_fourier(2 * q, eta, d)
    )
    if d != 2:
        assert float(py_dcf(1.5, eta, d, diameter=2.0)) < 0.0
    # edge: ρ c̃ depends on k only through kD
    rho2 = number_density(eta, d, 2.0)
    np.testing.assert_allclose(
        rho2 * bc_dcf_fourier(q / 2, eta, d, 2.0),
        number_density(eta, d) * bc_dcf_fourier(q, eta, d),
        rtol=1e-12,
        atol=1e-14,
    )
