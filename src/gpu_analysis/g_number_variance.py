#!/bin/bash
"""
Script for computing local number variance via an integral over the structure factor
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import os
import sys
import glob
import numpy as np
import numba as nb

# pylint: disable=E0401
sys.path.append("/home/sd2402/n_point_statistics")

from utils.geom import fsiv, trapz_integrate

from src.gpu_analysis.g_structure_factor import sample_xd_sf


def compute_2d_nv(sf: np.array, r_max: float = 8.0, delta_r: float = 5e-2) -> np.array:
    """
    DOCSTRING
    Args
        ...
    Returns
        ...
    """
    bins = np.arange(delta_r, r_max, delta_r)
    nv = np.zeros((bins.size, 2))

    nv[:, 0] = bins
    sk_to_nv_2d(nv, sf)

    return nv


@nb.njit(fastmath=True)
def sk_to_nv_2d(nv: np.array, sk: np.array):
    """
    Convert 2D S(k) data to number variance data via integration formula
    Assumes statistical isotropy so integration over
        k-vector can be done with simple polar transformation
    Args
        nv = (N, 2) ndarray of [radius, scaled_number_variance] for all radii
                to be populated with number variance data
        sk = (M, d+2) ndarray of structure factor data returned/computed from sampleXDsf()
    Returnss
        no return value, only POPULATES nv array
    Refs
        Original integral relation found in S.T. and F.S. (2003)
        https://torquatocpanel.deptcpanel.princeton.edu/wp-content/uploads/files/publications/papers/paper-211.pdf
    """
    k_vectors = sk[:, -2]
    structure_factors = sk[:, -1]
    radii = nv[:, 0]

    for ridx in range(radii.size):
        integrand = np.zeros_like(k_vectors)
        alphatilde = np.zeros_like(k_vectors)

        for kidx in range(k_vectors.size):
            alphatilde[kidx] = k_vectors[kidx] * fsiv(
                k_vectors[kidx], radii[ridx], 2
            )  # SPECIFIC TO 2D

            integrand[kidx] = (
                k_vectors[kidx] * fsiv(k_vectors[kidx], radii[ridx], 2) * structure_factors[kidx]
            )  # SPECIFIC TO 2D

        # if not (ridx % 10):
        #     I = 2*np.pi*trapz_integrate(K, alphatilde)
        #     unity = I/(2*np.pi)**2
        #     print("Expect that 1.0 ==", unity)

        nv[ridx, 1] = 2 * np.pi * trapz_integrate(k_vectors, integrand) / (2 * np.pi) ** 2
    return 0


def testing(idir: str, write_output: bool = False):
    """
    DOCSTRING
    Args
        idir = path to input directory of files w/ point patterns
                to compute the structure factor for
    """
    # pylint: disable=E0401
    sys.path.append("/home/sd2402/n-point_statistics")

    from utils.geom import angular_average
    from utils.readwrite import read_cco, write_xd_data

    # determine if we are using a directory of inputs (disordered system)
    # or just a single input (ordered system)
    if os.path.isdir(idir):
        configs = glob.glob("*__*", root_dir=idir)
    elif os.path.isfile(idir):
        idir = "".join(idir.split("/")[:-1]) + "./"
        configs = [idir]
    else:
        sys.exit("Cannot determine type of input (directory vs. single file)...")

    # metadata
    n_members = len(configs)
    eres = []  # ensemble averaging

    for midx in range(n_members):
        config = configs[midx]
        _, _, lv, coords = read_cco(idir + config)
        sfxv = sample_xd_sf(coords, lv)  # returns (W, d+2) ndarray
        eres.append(sfxv)

    eres = np.asarray(eres)
    ensemble_avg = np.mean(eres, axis=0)
    angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)

    # calculate number variance
    nv = compute_2d_nv(angular_avg, r_max=np.max(lv) / 8)

    if write_output:
        write_xd_data(nv, "nS_res.txt", header="Isotropic local number variance")

    return nv


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    CONFIG_DIR = sys.argv[1]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    res = testing(CONFIG_DIR, write_output=True)
