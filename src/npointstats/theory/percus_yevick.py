#!/usr/bin/env python3

"""
Percus–Yevick direct correlation function of hard rods (d = 1) and spheres (d = 3).

Closed-form PY solutions for hard particles of diameter D at packing fraction
η, with x = r/D:

    d = 1:  c(r) = -(1 - η x) / (1 - η)²                  (exact; Tonks)
    d = 3:  c(r) = -λ1 + 6η λ2 x - (η λ1 / 2) x³           (Wertheim, Thiele)

for r < D and c(r) = 0 for r > D, where λ1 = (1 + 2η)² / (1 - η)⁴ and
λ2 = (1 + η/2)² / (1 - η)⁴. The PY closure has no closed-form solution for
hard disks (d = 2); requesting it raises NotImplementedError.

Convention for the whole `npointstats.theory` subpackage: c̃(k) is returned
WITHOUT a density factor, c̃(k) = ∫ c(r) exp(-i k·r) d^d r, and
S(k) = 1 / (1 - ρ c̃(k)), ρ = η / v_1(D/2) the number density.

References
----------
J. K. Percus and G. J. Yevick, Phys. Rev. 110, 1 (1958).
M. S. Wertheim, Phys. Rev. Lett. 10, 321 (1963).
E. Thiele, J. Chem. Phys. 39, 474 (1963).
S. Torquato, Random Heterogeneous Materials (Springer, 2002).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import List

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike

# ----- local modules -----
from npointstats.theory.baus_colot import (
    _check_diameter,
    _check_eta,
    _truncate_at_core,
    number_density,
)
from npointstats.theory.hankel import (
    FloatArray,
    _as_radial_array,
    _check_dimension,
    ball_polynomial_transform,
)

# ============================================================
# COEFFICIENTS
# ============================================================


def _check_closure(d: int) -> None:
    """Raise ValueError for d outside {1, 2, 3}; NotImplementedError for d = 2."""
    _check_dimension(d)
    if d == 2:
        raise NotImplementedError(
            "the Percus-Yevick closure has no closed-form solution for hard disks "
            "(d = 2); use npointstats.theory.baus_colot for d = 2"
        )


def _py_coefficients(eta: float, d: int) -> List[float]:
    """Coefficients a_m of c(x) = Σ a_m x^m (x = r/D < 1) for d = 1 or 3."""
    if d == 1:
        scale = 1.0 / (1.0 - eta) ** 2
        return [-scale, eta * scale]
    lam1 = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    lam2 = (1.0 + 0.5 * eta) ** 2 / (1.0 - eta) ** 4
    return [-lam1, 6.0 * eta * lam2, 0.0, -0.5 * eta * lam1]


# ============================================================
# DIRECT CORRELATION FUNCTION
# ============================================================


def py_dcf(r: ArrayLike, eta: float, d: int, diameter: float = 1.0) -> FloatArray:
    """
    Percus–Yevick direct correlation function c(r) of hard rods or spheres.

    Parameters
    ----------
    r : array_like
        Radial distances r >= 0 (any shape).
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1 or 3.
    diameter : float, optional
        Particle diameter D > 0 (default 1).

    Returns
    -------
    ndarray
        c(r), same shape as r; zero for r > D and half its inner limit at r = D.

    Raises
    ------
    NotImplementedError
        If d = 2 (no closed-form PY solution for hard disks).
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    References
    ----------
    M. S. Wertheim, Phys. Rev. Lett. 10, 321 (1963).

    Examples
    --------
    >>> float(py_dcf(0.0, 0.5, 3))  # -(1 + 2η)²/(1 - η)⁴
    -64.0
    >>> float(py_dcf(2.0, 0.5, 3))
    0.0
    """
    _check_closure(d)
    value = _check_eta(eta, allow_zero=True)
    x = _as_radial_array(r, "r") / _check_diameter(diameter)
    inside = np.polynomial.polynomial.polyval(
        np.minimum(x, 1.0), _py_coefficients(value, d)
    )
    return _truncate_at_core(np.asarray(inside, dtype=np.float64), x)


def py_dcf_fourier(
    k: ArrayLike, eta: float, d: int, diameter: float = 1.0
) -> FloatArray:
    """
    Fourier transform c̃(k) = ∫ c(r) exp(-i k·r) d^d r of the PY DCF (exact).

    No density factor is included: S(k) = 1 / (1 - ρ c̃(k)). Evaluated in
    closed form, with a Taylor series for kD < 2 (no 0/0 at k = 0).

    Parameters
    ----------
    k : array_like
        Wavenumber magnitudes k >= 0 (any shape).
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1 or 3.
    diameter : float, optional
        Particle diameter D > 0 (default 1); c̃(k; D) = D^d c̃(kD; 1).

    Returns
    -------
    ndarray
        c̃(k), same shape as k, in units of D^d.

    Raises
    ------
    NotImplementedError
        If d = 2.
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    Examples
    --------
    >>> round(float(py_dcf_fourier(0.0, 0.0, 3)), 12) == round(-4 * np.pi / 3, 12)
    True
    """
    _check_closure(d)
    value = _check_eta(eta, allow_zero=True)
    sigma = _check_diameter(diameter)
    q = _as_radial_array(k, "k") * sigma
    coeffs = _py_coefficients(value, d)
    return np.asarray(sigma**d * ball_polynomial_transform(coeffs, q, d))


def py_structure_factor(
    k: ArrayLike, eta: float, d: int, diameter: float = 1.0
) -> FloatArray:
    """
    Percus–Yevick structure factor S(k) = 1 / (1 - ρ c̃(k)).

    Parameters
    ----------
    k : array_like
        Wavenumber magnitudes k >= 0 (any shape).
    eta : float
        Packing fraction, 0 <= eta < 1.
    d : int
        Dimension, 1 or 3.
    diameter : float, optional
        Particle diameter D > 0 (default 1).

    Returns
    -------
    ndarray
        S(k), same shape as k. S(0) = (1 - η)² in d = 1 and
        (1 - η)⁴ / (1 + 2η)² in d = 3 (compressibility route).

    Raises
    ------
    NotImplementedError
        If d = 2.
    ValueError
        If d is not 1, 2, 3, or any input is out of range.

    Examples
    --------
    >>> s0 = float(py_structure_factor(0.0, 0.4, 3))
    >>> round(s0, 12) == round(0.6**4 / 1.8**2, 12)
    True
    """
    _check_closure(d)
    ck = py_dcf_fourier(k, eta, d, diameter)
    rho = number_density(eta, d, diameter)
    return np.asarray(1.0 / (1.0 - rho * ck), dtype=np.float64)
