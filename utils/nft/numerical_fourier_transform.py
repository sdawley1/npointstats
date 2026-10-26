#!/bin/bash
"""
Numerical Fourier transform in 1-, 2-, and 3-dimensions
Assumes that we are transforming a function $F(r)$ that is statistically isotropic in $r$

References
    F. Lado. Numerical Fourier Transforms in One, Two, and Three Dimensions for Liquid State Calculations. J. Comp. Phys., 8 (1971).
    M. Baus and J. L. Colot. Thermodynamics and structure of a fluid of hard rods, disks, spheres, or hyperspheres from rescaled virial expansions. Phys. Rev. A., 36 (1987)

Sam Dawley
05/2025
"""

import sys

import numpy as np
from scipy.special import j0, j1, jn_zeros

# universal constants
PI = np.pi

# def orthogonality_1d(R: float, N: int) -> None:
#     # check orthogonality condition
#     delta_r = R / (N - 0.5)
#     delta_k = PI / R

#     for l in range(1, N-1):
#         r_l = (l - 0.5) * delta_r

#         for i in range(1, N - 1):
#             r_i = (i - 0.5) * delta_r

#             res = 0
#             for j in range(N - 1):
#                 k_j = (j - 0.5) * delta_k

#                 res += delta_k * np.cos(k_j * r_i) * np.cos(k_j * r_l)
#             if i == l:
#                 print(i, l, res * 2 * delta_r / PI)
#     return

# FORWARD FOURIER TRANSFORMS


def nft_1d(Fk: np.array, Fr: np.ufunc, R: float, N: int) -> None:
    """
    Forward Fourier transform in 1-dimension
    Args
        Fk = (M, 2) ndarray to be populated with Fourier transform of Fr
        Fr = np.ufunc of function to be transformed
        R = cutoff radius at which point F(r) = 0 for r >= R
        N = number of intervals to use
    """
    delta_r = R / (N - 0.5)
    delta_k = PI / R

    for j in range(1, Fk.shape[0] + 1):
        Fkj = 0
        k_j = (j - 0.5) * delta_k
        for i in range(1, N - 1):
            r_i = (i - 0.5) * delta_r
            Fkj += delta_r * Fr(r_i) * np.cos(k_j * r_i)
        Fk[j - 1] = [k_j, 2 * Fkj]

    # orthogonality_1d(R, N)
    return


def nft_2d(Fk: np.array, Fr: np.ufunc, R: float, N: int) -> None:
    """
    Forward Fourier transform in 1-dimension
    Args
        Fk = (M, 2) ndarray to be populated with Fourier transform of Fr
        Fr = np.ufunc of function to be transformed
        R = cutoff radius at which point F(r) = 0 for r >= R
        N = number of intervals to use
    """
    # order, zeros, and Nth zero of bessel function
    jzs = jn_zeros(0, max(Fk.shape[0], N))
    jzN = jzs[-1]
    K = jzN / R

    for j in range(1, Fk.shape[0] + 1):
        Fkj = 0
        k_j = jzs[j - 1] / R
        for i in range(1, N - 1):
            r_i = jzs[i - 1] * R / jzN
            Fkj += Fr(r_i) * j0(k_j * r_i) / j1(K * r_i) ** 2
        Fk[j - 1] = [k_j, 4 * PI * Fkj / K**2]
    return


def nft_3d(Fk: np.array, Fr: np.ufunc, R: float, N: int) -> None:
    """
    Forward Fourier transform in 1-dimension
    Args
        Fk = (M, 2) ndarray to be populated with Fourier transform of Fr
        Fr = np.ufunc of function to be transformed
        R = cutoff radius at which point F(r) = 0 for r >= R
        N = number of intervals to use
    """
    delta_r = R / N
    delta_k = PI / R

    for j in range(1, Fk.shape[0] + 1):
        Fkj = 0
        k_j = j * delta_k
        for i in range(1, N - 1):
            r_i = i * delta_r
            Fkj += delta_r * r_i * Fr(r_i) * np.sin(k_j * r_i)
        Fk[j - 1] = [k_j, 4 * PI * Fkj / k_j]
    return


# INVERSE FOURIER TRANSFORMS


def inft_1d(Fr: np.array, Fk: np.ufunc, R: float, N: int) -> None:
    """
    Inverse Fourier transform in 1-dimension
    Args
        Fr = (M, 2) ndarray to be populated with Fourier transform of Fr
        Fk = np.ufunc of function to be transformed
        R = cutoff radius at which point F(r) = 0 for r >= R
        N = number of intervals to use
    """
    delta_r = R / (N - 0.5)
    delta_k = PI / R

    for i in range(1, Fr.shape[0] + 1):
        Fri = 0
        r_i = (i - 0.5) * delta_r
        for j in range(1, N - 1):
            k_j = (j - 0.5) * delta_k
            Fri += delta_k * Fk(k_j) * np.cos(k_j * r_i)
        Fr[i - 1] = [r_i, Fri / PI]
    return


def inft_2d(Fr: np.array, Fk: np.ufunc, R: float, N: int) -> None:
    """
    Forward Fourier transform in 1-dimension
    Args
        Fk = (M, 2) ndarray to be populated with Fourier transform of Fr
        Fr = np.ufunc of function to be transformed
        R = cutoff radius at which point F(r) = 0 for r >= R
        N = number of intervals to use
    """
    # order, zeros, and Nth zero of bessel function
    jzs = jn_zeros(0, max(Fr.shape[0], N))
    jzN = jzs[-1]

    for i in range(1, Fr.shape[0] + 1):
        Fri = 0
        r_i = jzs[i - 1] * R / jzN
        for j in range(1, N - 1):
            k_j = jzs[j - 1] / R
            Fri += Fk(k_j) * j0(k_j * r_i) / j1(R * k_j) ** 2
        Fr[i - 1] = [r_i, Fri / PI / R**2]
    return


def inft_3d(Fr: np.array, Fk: np.ufunc, R: float, N: int) -> None:
    delta_r = R / N
    delta_k = PI / R

    for i in range(1, Fr.shape[0] + 1):
        Fri = 0
        r_i = i * delta_r
        for j in range(1, N - 1):
            k_j = j * delta_k
            Fri += delta_k * k_j * Fk(k_j) * np.sin(k_j * r_i)
        Fr[i - 1] = [r_i, Fri / 2 / PI**2 / r_i]
    return


# PRIMARY DRIVERS


def nft(f: np.ufunc, d: int, R: float, N: int) -> None:
    """
    Numerical Fourier transform in 1-, 2-, or 3-dimensions
    For cleaner (more resolved) plots, increase R
    Args
        f = function to Fourier transform
        d = dimensionality
        R = radius at which integrand vanishes for r >= R
        N = number of intervals to
    Returns
        Fk = (n_wavenumbers, 2) ndarray containing Fourier transform of f
    """
    n_wavenumbers = int(50 * R)  # tested empirically

    if N < n_wavenumbers:
        N = int(n_wavenumbers)

    Fk = np.zeros((n_wavenumbers, 2))  # statistical isotropy, I hardly know her!

    if d == 1:
        nft_1d(Fk, f, R, N)

    elif d == 2:
        nft_2d(Fk, f, R, N)

    elif d == 3:
        nft_3d(Fk, f, R, N)

    else:
        print(f"Dimension = {d:d} not supported...")
        sys.exit(1)

    return Fk


def inft(f: np.ufunc, d: int, R: float, N: int) -> None:
    """
    Inverse numerical Fourier transform in 1-, 2-, or 3-dimensions
    For cleaner (more resolved) plots, increase R
    Args
        f = function to Fourier transform
        d = dimensionality
        R = radius at which integrand vanishes for r >= R
        N = number of intervals to
    Returns
        Fr = (n_radii, 2) ndarray containing Fourier transform of f
    """
    n_radii = int(50 * R)  # tested empirically

    if N < n_radii:
        N = int(n_radii)

    Fr = np.zeros((n_radii, 2))  # statistical isotropy, I hardly know her!

    if d == 1:
        inft_1d(Fr, f, R, N)

    elif d == 2:
        inft_2d(Fr, f, R, N)

    elif d == 3:
        inft_3d(Fr, f, R, N)

    else:
        print(f"Dimension = {d:d} not supported...")
        sys.exit(1)

    return Fr


# TESTING


def PY(eta: float):
    # percus-yevick approximation to structure factor
    a1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    a2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4
    c = lambda x: (-a1 - 6 * eta * a2 * x - eta * a1 * x**3 / 2) * np.heaviside(1 - x, 0)
    return c(np.linspace(0, 5, 100))


def BC(eta: float):
    """
    Baus-Colot approximation to direct correlation function of a hard-sphere fluid
    """

    return


def testing():
    # metadata
    dimension = 3
    fourier_pfx = np.sqrt((2 * PI) ** dimension)

    # test functions
    fr_actual = lambda x: np.exp(-(x**2) / 2)
    fk_actual = lambda x: fourier_pfx * np.exp(-(x**2) / 2)

    # metadata
    cutoff_radius = 10
    num_intervals = 200

    # testing
    fr_test = inft(fk_actual, dimension, cutoff_radius, num_intervals)
    fk_test = nft(fr_actual, dimension, cutoff_radius, num_intervals)

    r_range = np.linspace(np.min(fr_test[:, 0]), np.max(fr_test[:, 0]), fr_test.shape[0])
    k_range = np.linspace(np.min(fk_test[:, 0]), np.max(fk_test[:, 0]), fk_test.shape[0])
    fr_comp = fr_actual(r_range)
    fk_comp = fk_actual(k_range)

    Es = []
    for a, b in zip(fk_comp, fk_test):
        E = np.abs(a - b[1]) * (2 / PI) ** (dimension / 2)
        Es.append(E)

    print("Forward Maximal error =", np.max(Es))

    Es = []
    for a, b in zip(fr_comp, fr_test):
        E = np.abs(a - b[1]) * (2 / PI) ** (dimension / 2)
        Es.append(E)

    print("Inverse Maximal error =", np.max(Es))

    return


def main():
    # testing()
    PY()

    return


if __name__ == "__main__":
    main()
