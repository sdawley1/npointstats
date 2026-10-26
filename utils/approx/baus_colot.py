# %%
#!/bin/bash
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
from scipy.optimize import fsolve
from scipy.special import jv

FIGSIZE = (7, 7)
FONTSIZE = 18
FIG_DPI = 512

# universal constants
# evaluating recurrence relation for compressibilities
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

# B_{N+1} / B_2^N
BN1OB2N = [
    # b_2, b_3, b_4, etc...
    [0.0],  # d=1
    [0.7820, 0.5322, 0.333_556_04, 0.198_83, 0.1148],
    [0.625, 0.2869, 0.110_252, 0.0389, 0.0137],
    [0.5063, 0.1513],
    [0.4140, 0.0746, 0.0148],
    [0.3409, 0.0328],
]


def read_file(inpfile: str):
    """Open and read-in lines from an input file"""
    with open(inpfile, "r", encoding="utf-8") as f:
        yield from f


def readXDdata(inpfile: str, idx: slice = slice(0, -1, 1), delim: str = ",") -> np.array:
    """
    Read data from inpfile of the form
        x_1,y_1,...,z_1
        x_2,y_2,...,z_2
        x_3,y_3,...,z_3, etc.
    Returns
        Z = (N, X) ndarray of data
    """
    header = ""
    Z = []
    for line in read_file(inpfile):
        if "#" in line:
            header += line
            continue
        L = line.strip("\n").split(delim)
        res = [float(a) for a in L[:]]
        Z.append(res)
    return np.asarray(Z)


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


def fcd3(xx: np.array, yy: np.array) -> np.array:
    """
    Third-order central finite difference to compute first derivative of yy wrt xx
    Use first-order forward, backward schemes at edges of the interval
    """
    yy_prime = np.zeros_like(xx)
    dX = np.ediff1d(xx)

    for i in range(yy.size):
        h = dX[i]
        if i == 0:
            dydx = (yy[i + 1] - yy[i]) / h
        elif i == yy.size - 1:
            dydx = (yy[i] - yy[i - 1]) / h
        else:
            dydx = (yy[i + 1] - yy[i - 1]) / 2 / h
        yy_prime[i] = dydx
    return yy_prime


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


# def rescaled_expansion_coefficient(d: int, n: int) -> np.array:
#     """
#     DOCSTRING
#     """
#     global BN1OB2N

#     # take care of general-dimensional explicit cases
#     if n < 0:
#         return 0.0
#     elif n == 0:
#         return 1.0
#     elif n == 1:
#         return 2**(d - 1) - d
#     elif n == 2:
#         return d * (d - 1) / 2 - d * math.pow(2, d - 1) + math.pow(4, d - 1) * BN1OB2N[d - 1][0]

#     # everything else applies when n >= 3

#     n_expansion_terms = len(BN1OB2N[d - 1])
#     expansion_virial_coefficients = lambda x: BN1OB2N[d - 1][x] # * math.pow(2, x * (d - 1)) *

#     # rescaled virial coefficients
#     # extra terms given by Baus, Colot ensure array is large enough
#     b = [1] + [expansion_virial_coefficients(i) for i in range(n_expansion_terms)] # [0] * (n - d) +
#     print("below, ", b)

#     if d == 2:
#         return b[n] - 2 * b[n - 1] + b[n - 2]

#     # c_n = 0
#     # for p in range(d):
#     #     bidx = n - p
#     #     if bidx < 0:
#     #         pass
#     #     elif bidx == 0:
#     #         c_n += (-1)**p / math.factorial(p) / math.factorial(d - p)
#     #     else:
#     #         c_n += (-1)**p / math.factorial(p) / math.factorial(d - p) * b[bidx]
#     # return c_n * math.factorial(d)


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


def bc_solve_a(d: int, eta_0: float, dZdeta: float) -> float:
    """
    Solve for parameter 'a' in Baus-Colot approximation; Necessary only for d >= 2
    Parameters:
    -----------
        d : int
            dimensionality
        eta_0 : float
            packing fraction
        dZdeta : float
            derivative of compressibility with respect to packing fraction evaluated at eta_0
    Returns:
    --------
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


def bc_dcf_1d(eta: float, r: float, q: float) -> tuple:
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


def bc_dcf_3d(eta: float, r: float, q: float, a: float, eta_dZdeta: float) -> tuple:
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
        eta_dZdeta : float
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
        * eta_dZdeta
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
        -eta_dZdeta
        * np.heaviside(1 - r, 0.5)
        * (1 + eta * r**3 / 2 - r * (2 * eta**2 + 8 * eta - 1 + 1 / eta_dZdeta) / 6 / eta)
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
) -> tuple:
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
            q = np.linspace(1e-2, 50, r.size)
        else:
            r = np.linspace(1e-2, 1.0, q.size)

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

    # print(f"sphere diameter D={D:0.4f}")
    # print(f"number density rho={rho:0.4f}")

    # start by getting EOS for a range of packing fractions around desired eta
    eta_range = np.linspace(eta - d_eta, eta + d_eta, n_eta)
    this_eta_idx = (np.abs(eta_range - eta)).argmin()
    Z = compressibility_recurrence(d, eta_range)
    print("Compressibility:", Z[this_eta_idx])

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


def main():
    dimension = 2
    number_density = 1.0
    K = np.linspace(1e-2, 50, 700)
    eta_max = np.pi / np.sqrt(12)  # 2d

    packing_fractions = eta_max / np.array([25, 10, 5, 4, 3, 2, 1.75, 1.5], dtype=np.float64)
    packing_fractions = np.array([0.54], dtype=np.float64)

    # temporary
    hdir = "/home/sd2402/scratch-sd2402/acoustic_kitty/"
    f1 = hdir + "rsa/RSA-2D-N_1000-rho_1.0_chi_0.2/initial_patterns/sf_bin.txt"
    tZ1 = readXDdata(f1)

    # print(f"d =", dimension)
    # for i in range(len(COMPRESSIBILITY_COEFFICIENTS[dimension - 1])):
    #     c_n = rescaled_expansion_coefficient(dimension, i)
    #     print("n =", i,"c_n =", c_n)
    #     print()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=512, layout="constrained")
    ax.set(xlim=[0, 1.5e1], ylim=[5e-2, 1e1])
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    # ax.set_xscale("log")
    ax.set_yscale("log")

    # xtick_labels = [0, 10, 20]
    # ytick_labels = [1e-2, 1e-1, 1e0, 1e1]
    # ax.set_ylim(8e-3, 8e0)

    # ax.set_xlim(xtick_labels[0], xtick_labels[-1])
    # ax.set_xticks(xtick_labels, labels=xtick_labels, fontsize=FONTSIZE)
    # ax.set_yticks(ytick_labels, labels=ytick_labels, fontsize=FONTSIZE)

    ax.set_xlabel(r"$k\sigma$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$S(k)$", fontsize=FONTSIZE)

    cmap = mpl.colormaps["jet"]
    cnorm = mpl.colors.Normalize(np.min(packing_fractions), np.max(packing_fractions))
    sg = dict(ls="-", lw=3)

    handles = []
    for pf in packing_fractions:
        rho = rho_from_pf(pf, dimension, D=1.0)
        Yr, Yk = bc_dcf(dimension, pf, q=K, rho=rho)
        S = 1 + number_density * Yk / (1 - number_density * Yk)
        print(f"eta={pf:0.3f}, S(k=0)={S[0]:3f}")

        # h, = ax.plot(K, Yr/2, label=rf"$\phi=${pf:0.2f}", **sg)
        # h, = ax.plot(K, S, label=f"{eta_max/pf:0.1f}", color=cmap(cnorm(pf)), **sg)
        (h,) = ax.plot(K, S, label=r"Eq. Hard Disks ($\phi=0.54$)", color=cmap(cnorm(pf)), **sg)
        handles.append(h)

    (h,) = ax.plot(tZ1[:, 0], tZ1[:, 1], **sg, label="Sat. RSA", color="r")
    handles.append(h)

    # bragg peaks
    x = 2 * np.pi * np.array([1, np.sqrt(2), 2, np.sqrt(5)])
    y = np.array([1, 1, 1, 2])
    for xc, yc in zip(x, y):
        ax.vlines(xc, 0, yc, color="blue", **sg)
    h = ax.hlines(1e-4, *ax.get_xlim(), color="blue", **sg, label="Square Lattice")
    handles.append(h)

    ax.legend(
        # title=r"$\phi_{max}/\phi$ " + rf"($d={dimension}$)",
        handles=handles,
        frameon=False,
        # ncols=2,
        title_fontsize=FONTSIZE,
        fontsize=FONTSIZE,
        loc="upper left",
    )

    plt.savefig(
        f"/home/sd2402/n_point_statistics/utils/approx/figures/baus_colot_{dimension:d}d.pdf",
        dpi=FIG_DPI,
    )

    return


# def tmp(p1: str, p2: str) -> None:

#     Z1 = readXDdata(p1)
#     Z2 = readXDdata(p2)

#     u_sg = dict(
#         lw = 4,
#         ls = "-"
#     )

#     fig, ax = plt.subplots(figsize = (7, 7), dpi = 512, layout="constrained")
#     ax.set(
#         yscale = "log",
#         xlim = [0, 10],
#         ylim = [1e-2, 5e0]
#     )

#     a_rsa = 2 * np.sqrt(0.54 / np.pi)
#     a_eqhs = 2 * np.sqrt(0.5 / np.pi)

#     ax.set_xlabel(r"$k\sigma$", fontsize=20)
#     ax.set_ylabel(r"$S(k)$", fontsize=20)

#     ax.plot(Z1[:, 0] * a_rsa, Z1[:, 1], color="r", label="Sat. RSA", **u_sg)
#     ax.plot(Z2[:, 0] * a_eqhs, Z2[:, 1], color="b", label=r"Eq. Hard Disks ($\phi=0.5$)", **u_sg)

#     # bragg peaks
#     x = 2 * np.pi * np.array([1, np.sqrt(2), 2, np.sqrt(5)])
#     y = np.array([1, 1, 1, 2])
#     for xc, yc in zip(x, y):
#         ax.vlines(xc, 0, yc, color="k", **u_sg)
#     ax.hlines(1e-4, *ax.get_xlim(), color="k", **u_sg, label="Square Lattice")

#     ax.legend(fontsize=18, frameon=False, loc="upper left")

#     plt.savefig("./tmp.pdf", dpi=512)

#     return

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    main()


# %%
