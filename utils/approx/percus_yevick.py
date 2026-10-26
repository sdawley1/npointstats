#!/bin/bash
"""
Percus-Yevick Approximation for the direct correlation function of hard sphere fluids

References
    M. Baus and J. L. Colot. Thermodynamics and structure of a fluid of hard rods, disks, spheres, or hyperspheres from rescaled virial expansions. Phys. Rev. A., 36 (1987)

Sam Dawley
05/2025
"""

import math
import sys

import numpy as np

sys.path.append("~/n_pt_statistics/")

from utils.nft import nft_1d


def py_dcf_1d(eta: float, r: float, q: float) -> tuple:
    """
    Percus-Yevick approximation to dcf of a 1D hard rod fluid
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
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    cr = -np.heaviside(1 - r, 0.5) * (1 - eta * r) / (1 - eta) ** 2

    ck = np.zeros((q.shape[0], 2))
    nft_1d(ck, cr, R=10, N=500)

    return cr, ck[:, 1]


def py_dcf_3d(eta: float, r: float, q: float) -> tuple:
    """
    Percus-Yevick approximation to dcf of a 3D hard sphere fluid
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
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    a1 = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    a2 = -((1 + eta / 2) ** 2) / (1 - eta) ** 4

    ck = (
        -4
        * np.pi
        * (
            a1 * (np.sin(q) - q * np.cos(q))
            + 6 * eta * a2 * (2 * q * np.sin(q) + (2 - q**2) * np.cos(q) - 2) / q
            + eta
            * a1
            * (4 * q * (q**2 - 6) * np.sin(q) - (24 - 12 * q**2 + q**4) * np.cos(q) + 24)
            / 2
            / q**3
        )
        / q**3
    )

    cr = -np.heaviside(1 - r, 0.5) * (a1 + 6 * eta * a2 * r + eta * a1 * r**3 / 2)
    return cr, ck


def py_dcf(
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
    Returns:
    --------
        cr : float, np.ndarray
            Real space direct correlation function
        ck : float, np.ndarray
            Fourier space direct correlation function
    """
    if d == 1:
        return py_dcf_1d(eta, q=q, r=r, D=D)
    elif d == 3:
        return py_dcf_3d(eta, q=q, r=r, D=D)


def main():
    return


if __name__ == "__main__":
    main()
