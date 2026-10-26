
# %%
"""
Baus-Colot Approximation for the direct correlation function of hard hypersphere fluids

References
    M. Baus and J. L. Colot. Thermodynamics and structure of a fluid of hard rods, disks, spheres, or hyperspheres from rescaled virial expansions. Phys. Rev. A., 36 (1987)

Sam Dawley
05/2025
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from scipy.special import jv

COMPRESSIBILITY_COEFFICIENTS = {
    1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [0.0, 0.1280, 0.0018, -0.0507, -0.0533, -0.0410],
    3: [1.0, 1.0, -0.6352, -0.8697, 0.2543, 2.9231],
    4: [4.0, 6.4057, -8.1170, 0.0, 0.0, 0.0],
    5: [11.0, 36.0, -74.438, 347.12, 0.0, 0.0],
}

# cf. Tables IV and V in Baus, Colot
Z6_RESULTS_2D = np.array(
    [
        [np.pi / 2 / np.sqrt(3) / x for x in [30, 20, 10, 5, 3, 2, 1.8, 1.6, 1.5, 1.4]],
        [
            1.063_44,
            1.097_54,
            1.210_68,
            1.498_42,
            2.077_21,
            3.424_52,
            4.171_93,
            5.498_13,
            6.614_33,
            8.352_33,
        ],
    ],
    dtype=np.float64,
)

Z6_RESULTS_3D = np.array(
    [
        [np.pi / 3 / np.sqrt(2) / x for x in [25, 18, 10, 5, 4, 3, 2, 1.8, 1.7, 1.6]],
        [
            1.127_750,
            1.182_839,
            1.359_427,
            1.888_516,
            2.244_506,
            3.033_017,
            5.877_755,
            7.495_420,
            8.706_121,
            10.373_926,
        ],
    ],
    dtype=np.float64,
)


def test_compressibility(d: int, d_eta: float = 0.1, n_eta: int = 500):
    """
    Test compressibility results against those of Baus, Colot for d=3 dimensions
    """
    global COMPRESSIBILITY_Z6_RESULTS

    if d == 2:
        packing_fractions = Z6_RESULTS_2D[0]
        compressibilities = Z6_RESULTS_2D[1]
    elif d == 3:
        packing_fractions = Z6_RESULTS_3D[0]
        compressibilities = Z6_RESULTS_3D[1]

    print(f"In d={d:d} dimensions")
    for idx, eta in enumerate(packing_fractions):
        eta_range = np.linspace(eta - d_eta, eta + d_eta, n_eta)
        this_eta_idx = (np.abs(eta_range - eta)).argmin()
        Z = compressibility_recurrence(d, eta_range)
        print(
            f"eta = {eta:.3f}, Z_BC = {compressibilities[idx]:>10.6f}, Z_me = {Z[this_eta_idx]:>10.6f}"
        )
    return


def rho_from_pf(eta: float, d: int, D: float) -> float:
    """
    Compute number density from packing fraction in d-dimensions
    """
    if d == 1:
        return eta / D
    elif d == 2:
        return np.sqrt(4 * eta / np.pi / D**2)
    elif d == 3:
        return np.cbrt(6 * eta / np.pi / D**3)


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
    delta_k = np.pi / R

    for j in range(1, Fk.shape[0] + 1):
        Fkj = 0
        k_j = (j - 0.5) * delta_k
        for i in range(1, N - 1):
            r_i = (i - 0.5) * delta_r
            Fkj += delta_r * Fr(r_i) * np.cos(k_j * r_i)
        Fk[j - 1] = [k_j, 2 * Fkj]

    return


def fcd5(xx: np.array, yy: np.array) -> np.array:
    """
    Fifth-order central finite difference to compute first derivative of yy wrt xx
    Use first-order forward, backward schemes at edges of the interval
    """
    yy_prime = np.zeros_like(xx)
    dX = np.ediff1d(xx)

    for i in range(yy.shape[0] - 1):
        h = dX[i]
        if i < 2:
            dydx = (yy[i + 1] - yy[i]) / h
        elif i >= yy.shape[0] - 2:
            dydx = (yy[i] - yy[i - 1]) / h
        else:
            dydx = (-yy[i + 2] + 8 * yy[i + 1] - 8 * yy[i - 1] + yy[i - 2]) / 12 / h
        yy_prime[i] = dydx
    return yy_prime


def bc_solve_a(d: int, eta_0: float, dZdeta: float) -> float:
    """
    Solve for parameter 'a' in Baus-Colot approximation
    Necessary only for d >= 2
    Args
        d = dimensionality
        eta_0 = packing fraction
        dZdeta = derivative of compressibility with respect to packing fraction evaluated at eta_0
    Returns
        ...
    """
    if d == 2:
        x, infodict, ier, mesg = fsolve(
            lambda x: (
                2
                * eta_0**2
                * (x**2 * (x**2 - 4) * np.arcsin(1 / x) - (x**2 + 2) * np.sqrt(x**2 - 1))
                - np.pi * (1 - 4 * eta_0 - 1 / dZdeta)
            ),
            x0=1.5,
            xtol=1e-12,
            fprime=lambda x: (
                (4 - x**2) / np.sqrt(1 - 1 / x**2)
                - x * (2 + x**2) / np.sqrt(x**2 - 1)
                - 2 * x * np.sqrt(x**2 - 1)
                - 4 * x * (2 - x**2) * np.arcsin(1 / x)
            ),
            full_output=True,
        )
    elif d == 3:
        x, infodict, ier, mesg = fsolve(
            lambda x: 9 * (eta_0 * x) ** 2 - 2 * eta_0**2 - 8 * eta_0 + 1 - 1 / dZdeta,
            x0=1.5,
            xtol=1e-12,
            fprime=lambda x: 2 * x,
            full_output=True,
        )
    return float(x[0]), infodict, ier, mesg


def compressibility_recurrence(d: int, eta: float) -> float:
    """
    Compute compressibility (dimensionless pressure) from recurrence relation of Baus and Colot
    Parameters:
    -----------
    d : float
        The exponent parameter d in (1-η)^{-d}
    eta : float
        The parameter η in the recurrence relation
    N : int
        Number of terms to include in the recurrence relation

    Returns:
    --------
    float
        The value of Z_N(η) after N iterations
    """
    global COMPRESSIBILITY_COEFFICIENTS
    N = len(COMPRESSIBILITY_COEFFICIENTS[d])

    # start at 1.0 to include base case N=0
    total = 1.0
    for n in range(N, 0, -1):
        total += COMPRESSIBILITY_COEFFICIENTS[d][n - 1] * eta**n
    return total / (1 - eta) ** d


def bc_dcf_1d(eta: float, r: float, q: float) -> float:
    """
    Baus-Colot approximation to the dcf in 1D
    Parameters:
    -----------
        eta : float
            packing fraction of spheres
        r : float, np.ndarray
            radius to compute dcf in real space
        q : float, np.ndarray
            wavenumber to compute dcf in Fourier space
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    f_one_half = lambda x: np.sin(x) / x
    ck = -eta * (2 * f_one_half(q) * (1 - eta) + eta * f_one_half(q / 2) ** 2) / (1 - eta) ** 2
    cr = -np.heaviside(1 - r, 0.5) * (1 - r * eta) / (1 - eta) ** 2
    return cr, ck


def bc_dcf_3d(eta: float, r: float, q: float, a: float, dZdeta: float) -> float:
    """
    Baus-Colot approximation to the dcf in 3D
    Parameters:
    -----------
        eta : float
            packing fraction of spheres
        r : float
            radius to compute dcf in real space
        q : float
            wavenumber to compute dcf in Fourier space
        a : float
            parameter 'a' from Baus-Colot approximation
        dZdeta : float
            derivative of compressibility wrt packing fraction evaluated at eta
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    ck = (
        -eta
        * dZdeta
        * (
            12
            * (
                24 * eta
                + 6 * eta * (a * q) ** 2
                + np.cos(q)
                * (
                    -24 * eta
                    + (12 * eta - 6 * eta * a**2) * q**2
                    - (2 + eta - 3 * eta * a**2) * q**4
                )
            )
            / q**6
            + 12 * np.sin(q) * (-24 * eta + (2 + 4 * eta - 6 * eta * a**2) * q**2) / q**5
        )
    )

    cr = (
        -dZdeta
        * np.heaviside(1 - r, 0.5)
        * (1 + eta * r**3 / 2 - r * (2 * eta**2 + 8 * eta - 1 + 1 / dZdeta) / 6 / eta)
    )

    return cr, ck


def bc_dcf(
    d: int,
    eta: float,
    r: float = 1.0,
    q: float = 2 * np.pi,
    D: float = 1.0,
    rho: float = None,
    d_eta: float = 0.2,
    n_eta: int = 200,
) -> float:
    """
    Parameters:
    -----------
        d : int
            dimension
        eta : float
            packing fraction of spheres
        r : float, np.ndarray
            radius to compute direct correlation function in real space
        q : float, np.ndarray
            wavenumber to compute direct correlation function in Fourier space
        D : float
            diameter of spheres
        rho : float
            number density of spheres
        d_eta : int
            interval around desired packing fraction to use for evaluating derivative dZ/d\eta
        n_eta : int
            number of terms to use in various calls to np.linspace()
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    if d > 3:
        print(f"Baus-Colot approximation not implemented for dimension = {d:d} > 3")
        sys.exit(1)

    # ensure r and q are of the appropriate type
    if type(r) is not type(q):
        if type(r) is np.ndarray:
            q = np.linspace(1e-3, 50, r.size)
        else:
            r = np.linspace(1e-3, 1.0, q.size)

    # return exact 1D result immediately
    if d == 1:
        return bc_dcf_1d(eta, r, q)

    # metadata
    do2 = d / 2
    if rho is None:
        rho = rho_from_pf(eta, d, D)
    else:
        rho_tmp = rho_from_pf(eta, d, D)
        if not np.isclose(rho, rho_tmp):
            print(
                f"Warning: packing fraction and density do not match! eta = {eta:0.3f}, rho = {rho:0.3f}, rho_tmp = {rho_tmp:0.3f}"
            )
            sys.exit(1)

    # start by getting EOS for a range of packing fractions around desired eta
    eta_range = np.linspace(eta - d_eta, eta + d_eta, n_eta)
    this_eta_idx = (np.abs(eta_range - eta)).argmin()
    Z = compressibility_recurrence(d, eta_range)

    eta_dZdeta = fcd5(eta_range, np.multiply(eta_range, Z))
    this_eta_dZdeta = eta_dZdeta[this_eta_idx]

    # get Baus-Colot paramters a and H_d
    a, infodict, ier, mesg = bc_solve_a(d, eta, this_eta_dZdeta)
    if not ier:
        print(f"Could not determine parameter 'a' for Baus-Colot approx.", flush=True)
        print(f"Result of root-finding:", infodict, flush=True)
        print(f"Error: {mesg}", flush=True)
        sys.exit(ier)

    # return exact 3D result as soon as possible
    if d == 3:
        return bc_dcf_3d(eta, r, q, a, this_eta_dZdeta)

    # everything beyond accounts for d=2 dimensions
    # nu = order, q = wavenumber
    f_nu = lambda nu, k: (2 / k) ** nu * math.gamma(1 + nu) * jv(nu, k)

    # compute auxiliary parameter H_d
    H_integration_range = np.linspace(1 / a, 1, n_eta)
    prefactor = 2 ** (d + 1) * math.gamma(1 + do2) / (math.gamma((d + 1) / 2) * math.gamma(1 / 2))
    integrand = lambda x, k: np.sqrt((1 - x**2) ** (d - 1)) * (
        f_nu(do2, k) - f_nu(do2, a * k * x) * (a * x) ** d
    )
    H_d = prefactor * np.array(
        [np.trapz(integrand(H_integration_range, kk), H_integration_range) for kk in q]
    )

    # finally! the dcf in Fourier space!
    ck = (
        -eta
        * this_eta_dZdeta
        * (
            2**d * (1 - eta * a**d) * f_nu(do2, q)
            + eta * a**d * ((f_nu(do2, a * q / 2) * a**do2) ** 2 + H_d)
        )
    )

    # also, the dcf in real space!
    cr = (
        -this_eta_dZdeta
        * np.heaviside(1 - r, 0.5)
        * (
            1
            - eta * a**2
            + 2 * eta * a**2 * (np.arccos(r / a) - r * np.sqrt(1 - (r / a) ** 2) / a) / np.pi
        )
    )

    return cr, ck


def py_dcf_tilde_1d(q: float, eta: float, D: float = 1.0) -> float:
    """
    DOCSTRING
    """
    cr = lambda r: -np.heaviside(1 - r, 0.5) * (1 - eta * r) / (1 - eta) ** 2

    ck = np.zeros((q.shape[0], 2))
    nft_1d(ck, cr, R=100, N=5000)

    return ck[:, 1]


def py_dcf_tilde_3d(q: float, eta: float, D: float = 1.0) -> float:
    """
    DOCSTRING
    """
    a1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    a2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4

    ck = (
        lambda kk: -4
        * np.pi
        * (
            a1 * (np.sin(kk) - kk * np.cos(kk))
            + 6 * eta * a2 * (2 * kk * np.sin(kk) + (2 - kk**2) * np.cos(kk) - 2) / kk
            + eta
            * a1
            * (4 * kk * (kk**2 - 6) * np.sin(kk) - (24 - 12 * kk**2 + kk**4) * np.cos(kk) + 24)
            / 2
            / kk**3
        )
        / kk**3
    )

    r = 1 / q
    cr = lambda x: -np.heaviside(1 - x, 0.5) * (a1 + 6 * eta * a2 * x + eta * a1 * x**3 / 2)
    return cr(r / D), ck(q * D)


def py_dcf_tilde(d: int, q: float, eta: float, D: float = 1.0) -> float:
    """
    Percus-Yevick appproximation to direct correlation function in Fourier space
    Assumes particles have unit diameter
    Args
        d = dimension
        q = wavenumber to compute direct correlation function in Fourier space
        eta = packing fraction of spheres
    """
    if d == 1:
        return py_dcf_tilde_1d(q, eta, D=D)
    elif d == 3:
        return py_dcf_tilde_3d(q, eta, D=D)


def testing(d: int) -> None:
    E = []

    eeta = np.linspace(1e-3, 0.9, 200)
    for ee in eeta:
        eta_range = np.linspace(ee - 0.2, ee + 0.2, 200)
        this_eta_idx = (np.abs(eta_range - ee)).argmin()
        Z = compressibility_recurrence(d, eta_range)

        eta_dZdeta = fcd5(eta_range, np.multiply(eta_range, Z))
        this_eta_dZdeta = eta_dZdeta[this_eta_idx]

        res, _, _, _ = bc_solve_a(d, ee, this_eta_dZdeta)
        E.append(res)
    E = np.asarray(E)

    # plotting
    fig, ax = plt.subplots(dpi=512)
    ax.set_xlim(0, 0.88)
    ax.set_ylim(1, 2)

    ax.plot(eeta, E)
    return


def main():
    dimension = 3
    diameter = 1e-2
    K = np.linspace(1e-2, 50, 500)
    R = np.linspace(1e-3, 1, 500)

    # testing(dimension)
    # return

    fig, ax = plt.subplots(dpi=512)
    ax.set_yscale("log")
    # ax.set_xlim(0, 1)
    ax.set_ylim(1e-2, 3e0)

    # ax.set_yticks([-i for i in range(28)])

    for pf in np.arange(0.0001, 0.001, 0.0001):
        rho = rho_from_pf(pf, dimension, D=diameter)
        Yr, Yk = bc_dcf(dimension, pf, q=K, rho=rho, D=diameter)
        S = 1 + rho * Yk / (1 - rho * Yk)

        # ax.plot(R, Yr, label=rf"BC $\eta=${pf:0.2f}")
        # ax.plot(K, Yk, label=rf"BC $\eta=${pf:0.2f}")
        ax.plot(K, S, label=rf"BC $\eta=${pf:0.2f}")

        if dimension != 2:
            print("number density =", rho)
            py_cr, py_ck = py_dcf_tilde(dimension, q=K, eta=pf, D=diameter)
            py_S = 1 + rho * py_ck / (1 - rho * py_ck)

            # ax.plot(diameter/K, py_cr, label=rf"PY $\eta=${pf:0.2f}", linestyle="--")
            # ax.plot(K, py_ck, label=rf"PY $\eta=${pf:0.2f}", linestyle="--")
            ax.plot(K, py_S, label=rf"PY $\eta=${pf:0.2f}", linestyle="--")

    # ax.legend()

    return


if __name__ == "__main__":
    main()


# %%
