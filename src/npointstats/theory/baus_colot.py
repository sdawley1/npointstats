#!/usr/bin/env python3

"""
Baus–Colot direct correlation function of hard rods, disks, and spheres.

Baus & Colot (1987) approximate the direct correlation function (DCF) of a
fluid of hard d-spheres of diameter D at packing fraction η by

    c(r) = -K [1 - η a^d + η a^d v_d(r / (a D))]   for r < D,   0 for r > D,

where v_d(x) is the normalized overlap volume of two d-spheres of unit
diameter at separation x, K = d(ηZ)/dη = β ∂P/∂ρ = 1/S(0) is the inverse
reduced isothermal compressibility of the rescaled-virial equation of state
Z(η), and the scale a is fixed by the compressibility sum rule
1 - ρ c̃(0) = K. In d = 1 the result is the exact Tonks DCF for any a.

Convention for the whole `npointstats.theory` subpackage: c̃(k) is returned
WITHOUT a density factor, c̃(k) = ∫ c(r) exp(-i k·r) d^d r, and
S(k) = 1 / (1 - ρ c̃(k)), ρ = η / v_1(D/2) the number density.

References
----------
M. Baus and J. L. Colot, Phys. Rev. A 36, 3912 (1987).
L. Tonks, Phys. Rev. 50, 955 (1936).
S. Torquato, Random Heterogeneous Materials (Springer, 2002).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Dict, Tuple

# ----- third party -----
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import ArrayLike
from scipy.special import j0, j1

# ----- local modules -----
from npointstats.theory.hankel import (
    FloatArray,
    _as_radial_array,
    _check_dimension,
    ball_polynomial_transform,
)

# ============================================================
# CONSTANTS
# ============================================================

# Numerators P_d(η) of the rescaled-virial equation of state
# Z(η) = P_d(η) / (1 - η)^d (Baus & Colot 1987, Z_6; d = 1 is exact Tonks).
# Coefficients are in increasing powers of η.
EOS_NUMERATORS: Dict[int, Tuple[float, ...]] = {
    1: (1.0,),
    2: (1.0, 0.0, 0.1280, 0.0018, -0.0507, -0.0533, -0.0410),
    3: (1.0, 1.0, 1.0, -0.6352, -0.8697, 0.2543, 2.9231),
}

# Safeguarded-Newton controls for the d = 2 scale parameter a.
A_INITIAL_GUESS = 2.0  # exact low-density limit (overlap of exclusion disks)
A_LOWER = 1.0  # the truncated-overlap residual is defined for a >= 1
A_UPPER = 3.0
A_XTOL = 1e-14
A_MAXITER = 100

# Gauss–Legendre panel rule for the d = 2 truncated-overlap integral.
GL_NODES_PER_PANEL = 32
GL_PHASE_PER_PANEL = 8.0
GL_MAX_KERNEL_ENTRIES = 4_000_000  # caps the (k, node) kernel block at 32 MB

# ============================================================
# STATE VARIABLES
# ============================================================


def _check_eta(eta: float, allow_zero: bool) -> float:
    """Return eta as float, raising ValueError unless 0 < eta < 1 (or 0 <=)."""
    if isinstance(eta, bool) or not isinstance(eta, (int, float, np.floating)):
        raise ValueError(f"packing fraction eta must be a real number, got {eta!r}")
    value = float(eta)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not (math.isfinite(value) and lower_ok and value < 1.0):
        bound = "0 <= eta < 1" if allow_zero else "0 < eta < 1"
        raise ValueError(f"packing fraction must satisfy {bound}, got {eta!r}")
    return value


def _check_diameter(diameter: float) -> float:
    """Return diameter as float, raising ValueError unless positive and finite."""
    if isinstance(diameter, bool) or not isinstance(
        diameter, (int, float, np.floating)
    ):
        raise ValueError(f"diameter must be a real number, got {diameter!r}")
    value = float(diameter)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"diameter must be positive and finite, got {diameter!r}")
    return value


def number_density(eta: float, d: int, diameter: float = 1.0) -> float:
    """
    Number density ρ = η / v_1(D/2) of hard d-spheres of diameter D.

    v_1(R) = π^{d/2} R^d / Γ(1 + d/2), so ρ = η/D (d = 1), 4η/(πD²) (d = 2),
    and 6η/(πD³) (d = 3).

    Parameters
    ----------
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1, 2, or 3.
    diameter : float, optional
        Sphere diameter D > 0 (default 1).

    Returns
    -------
    float
        Number density ρ.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, eta is outside [0, 1), or diameter <= 0.

    Examples
    --------
    >>> round(number_density(0.5, 3), 6)
    0.95493
    """
    _check_dimension(d)
    value = _check_eta(eta, allow_zero=True)
    sigma = _check_diameter(diameter)
    ball = np.pi ** (d / 2) * (sigma / 2.0) ** d / math.gamma(1.0 + d / 2.0)
    return float(value / ball)


def bc_compressibility_factor(eta: float, d: int) -> float:
    """
    Compressibility factor Z = βP/ρ of the Baus–Colot equation of state.

    Z(η) = P_d(η) / (1 - η)^d with the rescaled-virial numerators of Baus &
    Colot (1987): exact Tonks Z = 1/(1 - η) in d = 1; their Z_6 in d = 2, 3.

    Parameters
    ----------
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1, 2, or 3.

    Returns
    -------
    float
        Z(η).

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, or eta is outside [0, 1).

    Examples
    --------
    >>> round(bc_compressibility_factor(0.5, 1), 12)
    2.0
    """
    _check_dimension(d)
    value = _check_eta(eta, allow_zero=True)
    numerator = poly.polyval(value, EOS_NUMERATORS[int(d)])
    return float(numerator / (1.0 - value) ** d)


def bc_inverse_compressibility(eta: float, d: int) -> float:
    """
    Inverse reduced compressibility K = d(ηZ)/dη = β ∂P/∂ρ = 1/S(0), analytic.

    With ηZ = η P_d(η) (1 - η)^{-d},
    K = [(η P_d)' (1 - η) + d η P_d] / (1 - η)^{d+1}.

    Parameters
    ----------
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1, 2, or 3.

    Returns
    -------
    float
        K(η) >= 1.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, or eta is outside [0, 1).

    Examples
    --------
    >>> round(bc_inverse_compressibility(0.5, 1), 12)  # Tonks: 1/(1 - η)^2
    4.0
    """
    _check_dimension(d)
    value = _check_eta(eta, allow_zero=True)
    eta_p = poly.polymulx(np.asarray(EOS_NUMERATORS[int(d)], dtype=np.float64))
    term = poly.polyval(value, poly.polyder(eta_p)) * (1.0 - value)
    term += d * poly.polyval(value, eta_p)
    return float(term / (1.0 - value) ** (d + 1))


# ============================================================
# SCALE PARAMETER a
# ============================================================


def _disk_overlap(x: FloatArray) -> FloatArray:
    """Normalized overlap area v_2(x) of two unit-diameter disks, 0 <= x <= 1."""
    return np.asarray(
        (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(1.0 - x * x)), dtype=np.float64
    )


def _disk_overlap_moment(s: float) -> float:
    """P(s) = ∫_0^s x v_2(x) dx for 0 <= s <= 1 (closed form; P(1) = 1/8)."""
    root = math.sqrt(max(1.0 - s * s, 0.0))
    bracket = 0.5 * s * s * math.acos(s) + math.asin(s) / 8.0
    bracket -= s * (1.0 + 2.0 * s * s) * root / 8.0
    return (2.0 / math.pi) * bracket


def _bc_residual_2d(a: float, eta: float, k_inv: float) -> float:
    """
    Compressibility sum rule ρc̃(0) - (1 - K), divided by -K, for d = 2, a >= 1.

    R(a) = 4η [1 - η a² + η W(a)] - (1 - 1/K), with W(a) = 2 a^4 P(1/a) the
    disk-overlap integral truncated at r = D.
    """
    overlap = 2.0 * a**4 * _disk_overlap_moment(1.0 / a)
    return 4.0 * eta * (1.0 - eta * a * a + eta * overlap) - (1.0 - 1.0 / k_inv)


def _bc_jacobian_2d(a: float, eta: float) -> float:
    """Analytic dR/da of `_bc_residual_2d`: 4η² [-2a + 8a³P(1/a) - 2a v_2(1/a)]."""
    s = 1.0 / a
    v2 = float(_disk_overlap(np.asarray(s)))
    d_overlap = 8.0 * a**3 * _disk_overlap_moment(s) - 2.0 * a * v2
    return 4.0 * eta * eta * (-2.0 * a + d_overlap)


def _solve_a_2d(eta: float, k_inv: float) -> float:
    """
    Root of `_bc_residual_2d` in [A_LOWER, A_UPPER] by safeguarded Newton.

    Newton steps use the analytic Jacobian; a step leaving the current sign
    bracket is replaced by bisection, so a never drops below 1 (where the
    truncated-overlap form is undefined). Returns nan if there is no bracket.
    """
    lo, hi = A_LOWER, A_UPPER
    f_lo = _bc_residual_2d(lo, eta, k_inv)
    if f_lo * _bc_residual_2d(hi, eta, k_inv) > 0.0:
        return float("nan")
    x = A_INITIAL_GUESS
    for _ in range(A_MAXITER):
        f = _bc_residual_2d(x, eta, k_inv)
        if f == 0.0:
            return x
        if (f > 0.0) == (f_lo > 0.0):
            lo = x
        else:
            hi = x
        slope = _bc_jacobian_2d(x, eta)
        x_new = x - f / slope if slope != 0.0 else 0.5 * (lo + hi)
        if not lo < x_new < hi:
            x_new = 0.5 * (lo + hi)
        if abs(x_new - x) <= A_XTOL * max(1.0, abs(x)):
            return x_new
        x = x_new
    return float("nan")


def bc_solve_a(eta: float, d: int) -> float:
    """
    Scale parameter a of the Baus–Colot DCF from 1 - ρ c̃(0) = K.

    d = 3: closed form a = sqrt(2η² + 8η - 1 + 1/K) / (3η).
    d = 2: safeguarded Newton (analytic Jacobian) from the low-density limit 2.
    In d = 1 the DCF does not depend on a, so no value is defined.

    Parameters
    ----------
    eta : float
        Packing fraction, 0 < eta < 1.
    d : int
        Dimension, 2 or 3.

    Returns
    -------
    float
        a >= 1 (in units of D); a -> 2 as η -> 0.

    Raises
    ------
    ValueError
        If d is not 2 or 3, eta is outside (0, 1), or no root with a >= 1 exists.

    Examples
    --------
    >>> 1.0 < bc_solve_a(0.3, 3) < 2.0
    True
    """
    _check_dimension(d)
    if d == 1:
        raise ValueError("the Baus-Colot parameter a is undefined in d = 1")
    value = _check_eta(eta, allow_zero=False)
    k_inv = bc_inverse_compressibility(value, d)
    if d == 3:
        radicand = 2.0 * value**2 + 8.0 * value - 1.0 + 1.0 / k_inv
        a = math.sqrt(max(radicand, 0.0)) / (3.0 * value)
    else:
        a = _solve_a_2d(value, k_inv)
    if not (math.isfinite(a) and a >= 1.0):
        raise ValueError(f"no Baus-Colot root a >= 1 at eta = {value} in d = {d}")
    return a


# ============================================================
# DIRECT CORRELATION FUNCTION
# ============================================================


def _truncate_at_core(inside: FloatArray, x: FloatArray) -> FloatArray:
    """Multiply by θ(1 - x), taking the Fourier-midpoint value 1/2 at x = 1."""
    outside = np.where(x == 1.0, 0.5 * inside, 0.0)
    return np.asarray(np.where(x < 1.0, inside, outside), dtype=np.float64)


def bc_dcf(r: ArrayLike, eta: float, d: int, diameter: float = 1.0) -> FloatArray:
    """
    Baus–Colot direct correlation function c(r) of hard d-spheres.

    Parameters
    ----------
    r : array_like
        Radial distances r >= 0 (any shape).
    eta : float
        Packing fraction, 0 < eta < 1.
    d : int
        Dimension, 1, 2, or 3.
    diameter : float, optional
        Sphere diameter D > 0 (default 1).

    Returns
    -------
    ndarray
        c(r), same shape as r; zero for r > D and half its inner limit at r = D.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    References
    ----------
    M. Baus and J. L. Colot, Phys. Rev. A 36, 3912 (1987).

    Examples
    --------
    >>> float(bc_dcf(0.0, 0.5, 1))  # Tonks: -1/(1 - η)^2
    -4.0
    >>> float(bc_dcf(1.5, 0.3, 3))
    0.0
    """
    _check_dimension(d)
    value = _check_eta(eta, allow_zero=False)
    x = _as_radial_array(r, "r") / _check_diameter(diameter)
    xc = np.minimum(x, 1.0)
    if d == 1:
        inside = -(1.0 - value * xc) / (1.0 - value) ** 2
    else:
        k_inv = bc_inverse_compressibility(value, d)
        a = bc_solve_a(value, d)
        if d == 2:
            shape = 1.0 - value * a**2 * (1.0 - _disk_overlap(xc / a))
        else:
            shape = 1.0 - 1.5 * value * a**2 * xc + 0.5 * value * xc**3
        inside = -k_inv * shape
    return _truncate_at_core(inside, x)


def _jinc(q: FloatArray) -> FloatArray:
    """2 J_1(q) / q with its limit 1 at q = 0."""
    safe = np.where(q == 0.0, 1.0, q)
    return np.where(q == 0.0, 1.0, 2.0 * j1(safe) / safe)


def _truncated_overlap_2d(q: FloatArray, a: float) -> FloatArray:
    """
    T(q) = 2π ∫_D^{aD} x v_2(x/a) J_0(qx) dx (D = 1), by x = a cos θ.

    The substitution makes the integrand analytic in θ ∈ [0, arccos(1/a)];
    composite Gauss–Legendre panels keep ≤ 8 rad of J_0 phase per panel, and
    the k-axis is processed in blocks to bound memory.
    """
    theta_max = math.acos(1.0 / a)
    q_max = float(np.max(q)) if q.size else 0.0
    panels = 1 + int(q_max * (a - 1.0) / GL_PHASE_PER_PANEL)
    nodes, weights = np.polynomial.legendre.leggauss(GL_NODES_PER_PANEL)
    edges = np.linspace(0.0, theta_max, panels + 1)
    half = 0.5 * np.diff(edges)
    theta = (0.5 * (edges[:-1] + edges[1:]))[:, None] + half[:, None] * nodes
    w = (half[:, None] * weights).ravel()
    theta = theta.ravel()
    weighted = w * np.cos(theta) * np.sin(theta) * (2.0 * theta - np.sin(2.0 * theta))
    radius = a * np.cos(theta)
    out = np.empty(q.size, dtype=np.float64)
    rows = max(1, GL_MAX_KERNEL_ENTRIES // theta.size)
    for start in range(0, q.size, rows):
        block = q[start : start + rows]
        out[start : start + rows] = j0(np.outer(block, radius)) @ weighted
    return 2.0 * a * a * out


def bc_dcf_fourier(
    k: ArrayLike, eta: float, d: int, diameter: float = 1.0
) -> FloatArray:
    """
    Fourier transform c̃(k) = ∫ c(r) exp(-i k·r) d^d r of the Baus–Colot DCF.

    No density factor is included: S(k) = 1 / (1 - ρ c̃(k)). Exact in d = 1, 3;
    in d = 2 the truncated disk-overlap term is integrated by Gauss–Legendre
    quadrature (agrees with adaptive quadrature to ~1e-12).

    Parameters
    ----------
    k : array_like
        Wavenumber magnitudes k >= 0 (any shape).
    eta : float
        Packing fraction, 0 < eta < 1.
    d : int
        Dimension, 1, 2, or 3.
    diameter : float, optional
        Sphere diameter D > 0 (default 1); c̃(k; D) = D^d c̃(kD; 1).

    Returns
    -------
    ndarray
        c̃(k), same shape as k, in units of D^d.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    Examples
    --------
    >>> float(bc_dcf_fourier(0.0, 0.5, 1))  # -(2 - η)/(1 - η)^2
    -6.0
    """
    _check_dimension(d)
    value = _check_eta(eta, allow_zero=False)
    sigma = _check_diameter(diameter)
    q = _as_radial_array(k, "k") * sigma
    if d == 1:
        k_inv = 1.0 / (1.0 - value) ** 2
        return sigma * ball_polynomial_transform([-k_inv, value * k_inv], q, 1)
    k_inv = bc_inverse_compressibility(value, d)
    a = bc_solve_a(value, d)
    if d == 3:
        coeffs = [-k_inv, 1.5 * k_inv * value * a**2, 0.0, -0.5 * k_inv * value]
        return sigma**3 * ball_polynomial_transform(coeffs, q, 3)
    flat = q.ravel()
    overlap = 0.25 * np.pi * a**2 * _jinc(0.5 * a * flat) ** 2
    overlap -= _truncated_overlap_2d(flat, a)
    core = (1.0 - value * a**2) * np.pi * _jinc(flat)
    ck = -k_inv * (core + value * a**2 * overlap)
    return np.asarray(sigma**2 * ck.reshape(q.shape), dtype=np.float64)


def bc_structure_factor(
    k: ArrayLike, eta: float, d: int, diameter: float = 1.0
) -> FloatArray:
    """
    Baus–Colot structure factor S(k) = 1 / (1 - ρ c̃(k)).

    Parameters
    ----------
    k : array_like
        Wavenumber magnitudes k >= 0 (any shape).
    eta : float
        Packing fraction, 0 < eta < 1.
    d : int
        Dimension, 1, 2, or 3.
    diameter : float, optional
        Sphere diameter D > 0 (default 1).

    Returns
    -------
    ndarray
        S(k), same shape as k; S(0) = 1/K and S(k) -> 1 as k -> ∞.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    Examples
    --------
    >>> round(float(bc_structure_factor(0.0, 0.5, 1)), 12)  # Tonks: (1 - η)^2
    0.25
    """
    rho = number_density(eta, d, diameter)
    ck = bc_dcf_fourier(k, eta, d, diameter)
    return np.asarray(1.0 / (1.0 - rho * ck), dtype=np.float64)
