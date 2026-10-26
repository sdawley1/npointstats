#!/usr/bin/env python3

"""
Hypothesis property tests for `npointstats.theory` (Lado, PY, Baus–Colot).

Defect repros (D-034 … D-040) are pinned with `@example`.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Callable, Tuple

# ----- third party -----
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.integrate import quad
from scipy.special import j0, spherical_jn

# ----- local modules -----
from npointstats.theory import baus_colot as bc
from npointstats.theory import (
    ball_polynomial_transform,
    bc_compressibility_factor,
    bc_dcf,
    bc_dcf_fourier,
    bc_inverse_compressibility,
    bc_solve_a,
    bc_structure_factor,
    inverse_radial_fourier_transform,
    lado_grid,
    number_density,
    py_dcf,
    py_dcf_fourier,
    py_structure_factor,
    radial_fourier_transform,
)

# ============================================================
# STRATEGIES
# ============================================================

ETA_MAX = {1: 0.9, 2: 0.7, 3: 0.5}
finite = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}
dims = st.sampled_from([1, 2, 3])
odd_dims = st.sampled_from([1, 3])
diameters = st.floats(0.2, 5.0, **finite)
wavenumbers = st.floats(0.0, 40.0, **finite)
bad_dims = st.integers(-5, 0) | st.integers(4, 12)


@st.composite
def states(
    draw: st.DrawFn, d_strategy: st.SearchStrategy[int] = dims
) -> Tuple[int, float]:
    """Draw (d, η) with η in the physical fluid range for d."""
    d = draw(d_strategy)
    eta = draw(st.floats(0.01, ETA_MAX[d], **finite))
    return d, eta


@st.composite
def lado_samples(draw: st.DrawFn) -> Tuple[int, float, np.ndarray]:
    """Draw (d ∈ {1, 3}, r_max, samples) for an exact round trip."""
    d = draw(odd_dims)
    r_max = draw(st.floats(0.5, 50.0, **finite))
    n = draw(st.integers(2, 80))
    x = draw(hnp.arrays(np.float64, n, elements=st.floats(-1e3, 1e3, **finite)))
    return d, r_max, x


# ============================================================
# HELPERS
# ============================================================


def _quad_transform(c: Callable[[float], float], q: float, d: int) -> float:
    """c̃(q) = ∫ c(r) e^{-iq·r} d^d r over r ∈ [0, 1] by adaptive quadrature."""

    def integrand(r: float) -> float:
        if d == 1:
            return 2.0 * c(r) * np.cos(q * r)
        if d == 2:
            return 2.0 * np.pi * r * c(r) * float(j0(q * r))
        return 4.0 * np.pi * r * r * c(r) * float(spherical_jn(0, q * r))

    return quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=400)[0]


def _complex_step_k(eta: float, d: int) -> float:
    """Exact d(ηZ)/dη by complex step from the stored EOS numerators."""
    h = 1e-30
    z = eta + 1j * h
    p = np.polynomial.polynomial.polyval(z, bc.EOS_NUMERATORS[d])
    return float((z * p / (1.0 - z) ** d).imag / h)


# ============================================================
# LADO TRANSFORMS
# ============================================================


@given(lado_samples())
@example((1, 10.0, np.linspace(-1.0, 1.0, 40)))  # D-037: last node dropped
@example((3, 10.0, np.ones(40)))  # D-037
def test_lado_round_trip_is_identity_d1_d3(case: Tuple[int, float, np.ndarray]) -> None:
    d, r_max, x = case
    k, F = radial_fourier_transform(x, d, r_max, x.size)
    r, y = inverse_radial_fourier_transform(F, d, r_max, x.size)
    scale = max(1.0, float(np.abs(x).max()))
    np.testing.assert_allclose(y, x, rtol=0.0, atol=1e-10 * scale)
    assert k.shape == r.shape == x.shape


@given(dims, st.floats(0.3, 2.0, **finite))
def test_lado_gaussian_matches_analytic(d: int, sigma: float) -> None:
    r_max, n = 15.0 * sigma, 120
    k, F = radial_fourier_transform(
        lambda r: np.exp(-(r**2) / (2 * sigma**2)), d, r_max, n
    )
    exact = (2 * np.pi * sigma**2) ** (d / 2) * np.exp(-(sigma**2) * k**2 / 2)
    scale = (2 * np.pi * sigma**2) ** (d / 2)
    np.testing.assert_allclose(F, exact, rtol=0.0, atol=1e-10 * scale)
    _, y = inverse_radial_fourier_transform(F, d, r_max, n)
    r, _ = lado_grid(d, r_max, n)
    np.testing.assert_allclose(y, np.exp(-(r**2) / (2 * sigma**2)), atol=1e-10)


@given(dims, st.integers(2, 60), st.floats(-10.0, 10.0, **finite), st.data())
def test_lado_transform_is_linear(
    d: int, n: int, alpha: float, data: st.DataObject
) -> None:
    elems = st.floats(-10.0, 10.0, **finite)
    f = data.draw(hnp.arrays(np.float64, n, elements=elems))
    g = data.draw(hnp.arrays(np.float64, n, elements=elems))
    _, Ff = radial_fourier_transform(f, d, 5.0, n)
    _, Fg = radial_fourier_transform(g, d, 5.0, n)
    _, Fs = radial_fourier_transform(alpha * f + g, d, 5.0, n)
    scale = max(1.0, float(np.abs(Fs).max()), float(np.abs(alpha * Ff).max()))
    np.testing.assert_allclose(Fs, alpha * Ff + Fg, rtol=0.0, atol=1e-11 * scale)


@given(bad_dims, st.floats(0.1, 10.0, **finite))
def test_lado_rejects_unsupported_dimension(d: int, r_max: float) -> None:
    with pytest.raises(ValueError, match="dimension"):
        lado_grid(d, r_max, 8)


# ============================================================
# BALL POLYNOMIAL TRANSFORM
# ============================================================


@given(
    odd_dims,
    hnp.arrays(np.float64, st.integers(1, 6), elements=st.floats(-5.0, 5.0, **finite)),
    wavenumbers,
)
@example(3, np.array([1.0, -2.0, 0.0, 0.5]), 2.0)  # series/closed-form switch
@example(1, np.array([1.0, 1.0]), 0.0)
def test_ball_transform_matches_quadrature(
    d: int, coeffs: np.ndarray, k: float
) -> None:
    got = float(ball_polynomial_transform(coeffs, k, d))
    expected = _quad_transform(
        lambda x: float(np.polynomial.polynomial.polyval(x, coeffs)), k, d
    )
    scale = max(1.0, float(np.abs(coeffs).sum()))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-11 * scale)


@given(odd_dims, st.lists(st.floats(-5.0, 5.0, **finite), min_size=1, max_size=5),
       wavenumbers)  # fmt: skip
def test_ball_transform_is_even_in_k(d: int, coeffs: list, k: float) -> None:
    np.testing.assert_array_equal(
        ball_polynomial_transform(coeffs, k, d),
        ball_polynomial_transform(coeffs, -k, d),
    )


# ============================================================
# EQUATION OF STATE AND DENSITY
# ============================================================


@given(dims, st.floats(0.0, 0.95, **finite), diameters)
@example(3, 0.5, 1.0)  # D-034 repro: returned 0.985, expected 0.955
@example(2, 0.5, 1.0)  # D-034
def test_number_density_fills_eta(d: int, eta: float, diameter: float) -> None:
    rho = number_density(eta, d, diameter)
    volume = {1: diameter, 2: np.pi * diameter**2 / 4, 3: np.pi * diameter**3 / 6}[d]
    np.testing.assert_allclose(rho * volume, eta, rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(
        number_density(eta, d, 2 * diameter), rho / 2**d, rtol=1e-13, atol=1e-300
    )


@given(dims, st.floats(0.0, 0.9, **finite), st.floats(1e-4, 0.05, **finite))
def test_compressibility_factor_is_increasing_and_above_one(
    d: int, eta: float, step: float
) -> None:
    z = bc_compressibility_factor(eta, d)
    assert z >= 1.0
    assert bc_compressibility_factor(eta + step, d) > z
    if d == 1:
        np.testing.assert_allclose(z, 1.0 / (1.0 - eta), rtol=1e-14)


@given(dims, st.floats(0.0, 0.9, **finite))
@example(3, 0.45)  # D-036 repro: K evaluated at η ≈ 0.4498 by finite differences
@example(2, 0.85)  # D-036: the old FD grid ran past the pole at η = 1
def test_inverse_compressibility_is_exact_derivative(d: int, eta: float) -> None:
    k_inv = bc_inverse_compressibility(eta, d)
    np.testing.assert_allclose(k_inv, _complex_step_k(eta, d), rtol=1e-12)
    assert k_inv >= 1.0


# ============================================================
# BAUS–COLOT
# ============================================================


@given(states(st.sampled_from([2, 3])))
@example((2, 0.4))  # D-035 repro
@example((3, 0.4))  # D-035
def test_bc_a_solves_sum_rule_with_consistent_jacobian(
    state: Tuple[int, float],
) -> None:
    d, eta = state
    a = bc_solve_a(eta, d)
    assert 1.0 < a < 2.0
    k_inv = bc_inverse_compressibility(eta, d)
    if d == 3:
        residual = 9 * (eta * a) ** 2 - 2 * eta**2 - 8 * eta + 1 - 1 / k_inv
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        return
    np.testing.assert_allclose(bc._bc_residual_2d(a, eta, k_inv), 0.0, atol=1e-13)
    for x in (1.1, a, 1.9):
        h = 1e-6
        fd = bc._bc_residual_2d(x + h, eta, k_inv) - bc._bc_residual_2d(
            x - h, eta, k_inv
        )
        jac = bc._bc_jacobian_2d(x, eta)
        np.testing.assert_allclose(jac, fd / (2 * h), rtol=1e-5, atol=1e-9 * eta**2)


@given(states())
@example((3, 0.45))
def test_bc_reproduces_its_compressibility(state: Tuple[int, float]) -> None:
    d, eta = state
    rho = number_density(eta, d)
    k_inv = bc_inverse_compressibility(eta, d)
    lhs = 1.0 - rho * float(bc_dcf_fourier(0.0, eta, d))
    np.testing.assert_allclose(lhs, k_inv, rtol=1e-10)
    np.testing.assert_allclose(
        float(bc_structure_factor(0.0, eta, d)), 1.0 / k_inv, rtol=1e-10
    )


@given(states(), wavenumbers)
def test_bc_fourier_matches_quadrature(state: Tuple[int, float], q: float) -> None:
    d, eta = state
    expected = _quad_transform(lambda r: float(bc_dcf(r, eta, d)), q, d)
    got = float(bc_dcf_fourier(q, eta, d))
    scale = max(1.0, abs(float(bc_dcf(0.0, eta, d))))
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-10 * scale)


@given(states(), diameters, st.floats(1.0001, 10.0, **finite))
def test_bc_dcf_vanishes_outside_core(
    state: Tuple[int, float], diameter: float, ratio: float
) -> None:
    d, eta = state
    assert float(bc_dcf(ratio * diameter, eta, d, diameter)) == pytest.approx(
        0.0, abs=0.0
    )
    assert float(bc_dcf(0.0, eta, d, diameter)) < 0.0


@given(states(), diameters, wavenumbers)
def test_bc_diameter_scaling(
    state: Tuple[int, float], diameter: float, q: float
) -> None:
    d, eta = state
    scaled = float(bc_dcf_fourier(q / diameter, eta, d, diameter))
    unit = float(bc_dcf_fourier(q, eta, d))
    np.testing.assert_allclose(
        scaled, diameter**d * unit, rtol=1e-12, atol=1e-12 * diameter**d
    )


@given(states(), st.floats(2e3, 5e3, **finite))
def test_bc_structure_factor_tends_to_one(state: Tuple[int, float], k: float) -> None:
    d, eta = state
    assert abs(float(bc_structure_factor(k, eta, d)) - 1.0) < 0.05


# ============================================================
# PERCUS–YEVICK
# ============================================================


@given(st.floats(0.0, 0.5, **finite))
def test_py_3d_s0_compressibility_route(eta: float) -> None:
    expected = (1 - eta) ** 4 / (1 + 2 * eta) ** 2
    np.testing.assert_allclose(
        float(py_structure_factor(0.0, eta, 3)), expected, rtol=1e-12
    )


@given(st.floats(0.0, 0.9, **finite), wavenumbers)
@example(0.4, 0.5)  # D-039 repro: BC 1D returned ρc̃ (factor η too small)
def test_py_and_bc_agree_with_tonks_in_1d(eta: float, q: float) -> None:
    np.testing.assert_allclose(
        float(py_structure_factor(0.0, eta, 1)), (1 - eta) ** 2, rtol=1e-12
    )
    if eta > 0.0:
        np.testing.assert_allclose(
            float(bc_dcf_fourier(q, eta, 1)), float(py_dcf_fourier(q, eta, 1)),
            rtol=1e-12, atol=1e-12 / (1 - eta) ** 2,
        )  # fmt: skip


@given(states(odd_dims), wavenumbers)
def test_py_fourier_matches_quadrature(state: Tuple[int, float], q: float) -> None:
    d, eta = state
    expected = _quad_transform(lambda r: float(py_dcf(r, eta, d)), q, d)
    scale = max(1.0, abs(float(py_dcf(0.0, eta, d))))
    np.testing.assert_allclose(float(py_dcf_fourier(q, eta, d)), expected, rtol=1e-9,
                               atol=1e-10 * scale)  # fmt: skip


@given(states(odd_dims), diameters, st.floats(1.0001, 10.0, **finite))
def test_py_dcf_vanishes_outside_core(
    state: Tuple[int, float], diameter: float, ratio: float
) -> None:
    d, eta = state
    assert float(py_dcf(ratio * diameter, eta, d, diameter)) == pytest.approx(
        0.0, abs=0.0
    )


@given(states(odd_dims), st.floats(2e3, 1e5, **finite))
def test_py_structure_factor_tends_to_one(state: Tuple[int, float], k: float) -> None:
    d, eta = state
    assert abs(float(py_structure_factor(k, eta, d)) - 1.0) < 0.05


@given(st.floats(0.0, 0.7, **finite), st.floats(0.0, 3.0, **finite))
@example(0.3, 0.5)  # D-038 repro: d = 2 returned None
def test_py_2d_always_raises(eta: float, r: float) -> None:
    with pytest.raises(NotImplementedError):
        py_dcf(r, eta, 2)
    with pytest.raises(NotImplementedError):
        py_structure_factor(r, eta, 2)


# ============================================================
# DISPATCH CONTRACT
# ============================================================


@given(bad_dims, st.floats(0.01, 0.5, **finite))
@example(4, 0.5)  # D-040 repro: Approx.apx_dcf returned 1 for d = 4
def test_unsupported_dimension_raises_for_every_closure(d: int, eta: float) -> None:
    for func in (bc_dcf, bc_dcf_fourier, bc_structure_factor, py_dcf, py_dcf_fourier,
                 py_structure_factor):  # fmt: skip
        with pytest.raises(ValueError, match="dimension"):
            func(0.5, eta, d)
