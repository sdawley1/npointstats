#!/usr/bin/env python3

"""
Radial (Hankel-type) Fourier transforms of isotropic functions in d = 1, 2, 3.

Discrete forward and inverse transforms on the grids of Lado (1971), and the
closed-form transform of a polynomial supported on the unit ball (used for the
direct correlation functions of hard rods and spheres).

Convention (Torquato's): for a radial function f(r) in d dimensions,

    F(k) = ∫ f(r) exp(-i k·r) d^d r,
    f(r) = (2π)^{-d} ∫ F(k) exp(i k·r) d^d k.

Lado's N is `n + 1` here: every grid holds exactly `n` nodes, i = 1 … N - 1.
In d = 1 and d = 3 the discrete pair is exactly orthogonal, so a forward then
inverse transform returns any input to round-off. In d = 2 (Fourier–Bessel
nodes at zeros of J0) orthogonality is only approximate; the round trip is
accurate for functions that are negligible beyond `r_max` and band-limited
below the largest wavenumber.

References
----------
F. Lado, J. Comput. Phys. 8, 417 (1971).
S. Torquato, Random Heterogeneous Materials (Springer, 2002).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from typing import Callable, Tuple, Union

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0, j1, jn_zeros

# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_DIMENSIONS = (1, 2, 3)

# Below this |k| the ball transform uses its Taylor series (the closed form
# suffers catastrophic cancellation as k -> 0); above it, the recurrence.
SERIES_SWITCH_K = 2.0

# Number of Taylor terms; (2^2)^30 / 60! < 1e-63, far below round-off.
SERIES_TERMS = 30

FloatArray = NDArray[np.float64]
RadialInput = Union[Callable[[FloatArray], ArrayLike], ArrayLike]

# ============================================================
# VALIDATION
# ============================================================


def _check_dimension(d: int) -> None:
    """Raise ValueError unless d is an integer in {1, 2, 3}."""
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise ValueError(f"dimension d must be an integer, got {d!r}")
    if int(d) not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension d must be 1, 2, or 3, got {d}")


def _check_grid(r_max: float, n: int) -> None:
    """Raise ValueError unless r_max > 0 is finite and n >= 2 is an integer."""
    if not (isinstance(r_max, (int, float, np.floating)) and math.isfinite(r_max)):
        raise ValueError(f"r_max must be a finite real number, got {r_max!r}")
    if r_max <= 0.0:
        raise ValueError(f"r_max must be positive, got {r_max}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer, got {n!r}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")


def _sample(f: RadialInput, x: FloatArray, name: str) -> FloatArray:
    """Evaluate a callable on grid x, or validate pre-sampled values against x."""
    values = f(x) if callable(f) else f
    out = np.array(values, dtype=np.float64)
    if out.shape != x.shape:
        raise ValueError(
            f"{name} must have shape {x.shape} on the Lado grid, got {out.shape}"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _as_radial_array(x: ArrayLike, name: str) -> FloatArray:
    """Return x as a float array, raising ValueError unless finite and >= 0."""
    out = np.array(x, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if np.any(out < 0.0):
        raise ValueError(f"{name} must be non-negative (it is a radial magnitude)")
    return out


# ============================================================
# LADO GRIDS
# ============================================================


def lado_grid(d: int, r_max: float, n: int) -> Tuple[FloatArray, FloatArray]:
    """
    Real-space and Fourier-space nodes of Lado's discrete radial transform.

    Parameters
    ----------
    d : int
        Dimension, 1, 2, or 3.
    r_max : float
        Cutoff radius R; the transformed function is taken to vanish for r >= R.
    n : int
        Number of nodes (Lado's N - 1), at least 2.

    Returns
    -------
    r : ndarray, shape (n,)
        Radii. d = 1: (i - 1/2) R / (n + 1/2); d = 2: j_{0,i} R / j_{0,n+1};
        d = 3: i R / (n + 1).
    k : ndarray, shape (n,)
        Wavenumbers. d = 1: (j - 1/2) π / R; d = 2: j_{0,j} / R; d = 3: j π / R.

    Raises
    ------
    ValueError
        If d is not 1, 2, 3, r_max is not positive and finite, or n < 2.

    References
    ----------
    F. Lado, J. Comput. Phys. 8, 417 (1971), Eqs. (9), (17), (23).

    Examples
    --------
    >>> r, k = lado_grid(3, 10.0, 4)
    >>> r.tolist(), (k * 10.0 / np.pi).round(12).tolist()
    ([2.0, 4.0, 6.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    """
    _check_dimension(d)
    _check_grid(r_max, n)
    index = np.arange(1, n + 1, dtype=np.float64)
    if d == 1:
        r = (index - 0.5) * r_max / (n + 0.5)
        k = (index - 0.5) * np.pi / r_max
    elif d == 2:
        zeros = np.asarray(jn_zeros(0, n + 1), dtype=np.float64)
        r = zeros[:n] * r_max / zeros[n]
        k = zeros[:n] / r_max
    else:
        r = index * r_max / (n + 1)
        k = index * np.pi / r_max
    return r, k


# ============================================================
# DISCRETE TRANSFORMS
# ============================================================


def _lado_matrices(d: int, r_max: float, n: int) -> Tuple[FloatArray, FloatArray]:
    """Forward (k <- r) and inverse (r <- k) Lado matrices, both shape (n, n)."""
    r, k = lado_grid(d, r_max, n)
    kr = np.outer(k, r)
    if d == 1:
        dr = r_max / (n + 0.5)
        kernel = np.cos(kr)
        forward = 2.0 * dr * kernel
        inverse = kernel.T / r_max
    elif d == 2:
        zeros = np.asarray(jn_zeros(0, n + 1), dtype=np.float64)
        weights = 1.0 / j1(zeros[:n]) ** 2
        k_top = zeros[n] / r_max
        kernel = j0(kr)
        forward = (4.0 * np.pi / k_top**2) * kernel * weights[np.newaxis, :]
        inverse = kernel.T * weights[np.newaxis, :] / (np.pi * r_max**2)
    else:
        dr = r_max / (n + 1)
        kernel = np.sin(kr)
        forward = 4.0 * np.pi * dr * kernel * (r[np.newaxis, :] / k[:, np.newaxis])
        dk = np.pi / r_max
        inverse = dk * kernel.T * (k[np.newaxis, :] / r[:, np.newaxis]) / (2 * np.pi**2)
    return forward, inverse


def radial_fourier_transform(
    f: RadialInput, d: int, r_max: float, n: int
) -> Tuple[FloatArray, FloatArray]:
    """
    Forward radial Fourier transform F(k) = ∫ f(r) exp(-i k·r) d^d r (Lado 1971).

    Parameters
    ----------
    f : callable or array_like
        Either a vectorized function f(r) evaluated on the Lado r-grid, or its
        n samples on that grid (see `lado_grid`).
    d : int
        Dimension, 1, 2, or 3.
    r_max : float
        Cutoff radius R; f must vanish (or be negligible) for r >= R.
    n : int
        Number of nodes, at least 2. Cost is O(n^2) time and memory.

    Returns
    -------
    k : ndarray, shape (n,)
        Lado wavenumbers.
    F : ndarray, shape (n,)
        Transform at k.

    Raises
    ------
    ValueError
        On invalid d, r_max, n, wrong sample shape, or non-finite samples.

    References
    ----------
    F. Lado, J. Comput. Phys. 8, 417 (1971).

    Examples
    --------
    >>> k, F = radial_fourier_transform(lambda r: np.exp(-r**2 / 2), 3, 12.0, 64)
    >>> bool(np.allclose(F, (2 * np.pi) ** 1.5 * np.exp(-k**2 / 2), atol=1e-12))
    True
    """
    _check_dimension(d)
    r, k = lado_grid(d, r_max, n)
    samples = _sample(f, r, "f")
    forward, _ = _lado_matrices(int(d), float(r_max), int(n))
    return k, forward @ samples


def inverse_radial_fourier_transform(
    F: RadialInput, d: int, r_max: float, n: int
) -> Tuple[FloatArray, FloatArray]:
    """
    Inverse radial transform f(r) = (2π)^{-d} ∫ F(k) exp(i k·r) d^d k (Lado 1971).

    Parameters
    ----------
    F : callable or array_like
        Either a vectorized function F(k) evaluated on the Lado k-grid, or its
        n samples on that grid (see `lado_grid`).
    d : int
        Dimension, 1, 2, or 3.
    r_max : float
        Real-space cutoff radius R that fixes both grids.
    n : int
        Number of nodes, at least 2.

    Returns
    -------
    r : ndarray, shape (n,)
        Lado radii.
    f : ndarray, shape (n,)
        Inverse transform at r.

    Raises
    ------
    ValueError
        On invalid d, r_max, n, wrong sample shape, or non-finite samples.

    Examples
    --------
    >>> g = lambda k: np.sqrt(2 * np.pi) * np.exp(-k**2 / 2)
    >>> r, f = inverse_radial_fourier_transform(g, 1, 12.0, 64)
    >>> bool(np.allclose(f, np.exp(-r**2 / 2), atol=1e-12))
    True
    """
    _check_dimension(d)
    r, k = lado_grid(d, r_max, n)
    samples = _sample(F, k, "F")
    _, inverse = _lado_matrices(int(d), float(r_max), int(n))
    return r, inverse @ samples


# ============================================================
# ANALYTIC TRANSFORM OF A BALL POLYNOMIAL
# ============================================================


def _moments_series(q: FloatArray, degree: int, d: int) -> FloatArray:
    """Taylor series of the radial moments M_m(q), m = 0 … degree (see below)."""
    ell = np.arange(SERIES_TERMS, dtype=np.float64)
    shift = 0.0 if d == 1 else 1.0
    factorial = np.array(
        [math.factorial(int(2 * j + shift)) for j in ell], dtype=np.float64
    )
    powers = (-(q[:, np.newaxis] ** 2)) ** ell[np.newaxis, :]
    out = np.empty((degree + 1, q.size), dtype=np.float64)
    for m in range(degree + 1):
        denom = factorial * (m + 2 * ell + 1 + 2 * shift)
        out[m] = powers @ (1.0 / denom)
    return out


def _moments_closed(q: FloatArray, degree: int, d: int) -> FloatArray:
    """Closed-form moments from I_m(q) = ∫_0^1 x^m e^{iqx} dx by recurrence."""
    phase = np.exp(1j * q)
    top = degree + (0 if d == 1 else 1)
    current = (phase - 1.0) / (1j * q)
    exact = np.empty((top + 1, q.size), dtype=np.complex128)
    exact[0] = current
    for m in range(1, top + 1):
        current = (phase - m * current) / (1j * q)
        exact[m] = current
    if d == 1:
        return np.asarray(exact.real, dtype=np.float64)
    return np.asarray(exact[1:].imag / q[np.newaxis, :], dtype=np.float64)


def ball_polynomial_transform(coeffs: ArrayLike, k: ArrayLike, d: int) -> FloatArray:
    """
    Fourier transform of p(|x|) = Σ_m a_m |x|^m restricted to the unit ball.

    Computes F(k) = ∫_{|x|<1} p(|x|) exp(-i k·x) d^d x exactly, i.e.
    d = 1: 2 Σ a_m ∫_0^1 x^m cos(kx) dx;  d = 3: 4π Σ a_m ∫_0^1 x^{m+2} j_0(kx) dx.
    A Taylor series is used for |k| < 2 to avoid the 0/0 cancellation of the
    closed form; the integration-by-parts recurrence is used above.

    Parameters
    ----------
    coeffs : array_like, shape (m_max + 1,)
        Polynomial coefficients a_0, a_1, …, in increasing powers of |x|.
    k : array_like
        Wavenumbers (any shape; the transform is even in k).
    d : int
        Dimension, 1 or 3.

    Returns
    -------
    ndarray
        F(k), same shape as `k`.

    Raises
    ------
    ValueError
        If d is not 1 or 3, coeffs is empty or not 1-D, or inputs are non-finite.

    Examples
    --------
    >>> float(ball_polynomial_transform([1.0], 0.0, 3)) / (4 * np.pi / 3)
    1.0
    >>> round(float(ball_polynomial_transform([1.0], np.pi, 1)), 14) + 0.0
    0.0
    """
    if d not in (1, 3) or isinstance(d, bool):
        raise ValueError(f"ball_polynomial_transform supports d = 1 or 3, got {d!r}")
    a = np.asarray(coeffs, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("coeffs must be a non-empty finite 1-D array")
    kk = np.abs(np.asarray(k, dtype=np.float64))
    if not np.all(np.isfinite(kk)):
        raise ValueError("k must be finite")
    q = kk.ravel()
    degree = a.size - 1
    small = q < SERIES_SWITCH_K
    moments = np.empty((degree + 1, q.size), dtype=np.float64)
    moments[:, small] = _moments_series(q[small], degree, d)
    moments[:, ~small] = _moments_closed(q[~small], degree, d)
    prefactor = 2.0 if d == 1 else 4.0 * np.pi
    return np.asarray(prefactor * (a @ moments), dtype=np.float64).reshape(kk.shape)
