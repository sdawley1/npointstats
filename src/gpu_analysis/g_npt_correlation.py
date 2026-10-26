#!/bin/bash
"""
Script for sampling n-pt correlation functions
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import os
import sys
import glob
import time
import numpy as np

# pylint: disable=E0401
sys.path.append("/home/sd2402/n_point_statistics")

from src.gpu_analysis.g_pairwise_distances import dist_matrix_xd

from utils.geom import parallelipiped_volume, shell_volume


def sample_xd_g2(
    point_pattern: np.array, lv: np.array, tpb: tuple = (32, 32), delta_r: float = 2e-2
) -> None:
    """
    Sample d-dimensional pair correlation function g2(r)
    Args
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        lv = (d, d) ndarray of primitive lattice vectors
        tpb = threads per block to use for GPU
        delta_r = spacing between successive radii for binning result
    Returns
        sfxv = (W, d+2) ndarray w/ elements [k_{x1}, ..., k_{xd}, |k|, S(k)]
            for all W wave vectors
    """

    # derived metadata
    n_particles, dimension = point_pattern.shape
    vol = parallelipiped_volume(lv)
    rho = n_particles / vol
    delta_r = 1e-3 / vol ** (1 / dimension)

    bins = np.arange(delta_r, np.max(lv) / 2, delta_r)
    rdiffs = np.ediff1d(bins)

    g2 = np.zeros((bins.size - 1, 2))
    radii = dist_matrix_xd(point_pattern, point_pattern, lv, tpb=tpb, self_dist=0.0)
    hist, _ = np.histogram(radii.ravel(), bins)

    g2[:, 0] = bins[1:]
    g2[:, 1] = np.multiply(
        hist, 1 / shell_volume(bins[1:], dimension, rdiffs) / rho / n_particles
    )  # fixed in 2D

    return g2


def testing(idir: str, write_output: bool = False):
    """
    DOCSTRING
    Args
        idir = path to input directory containing files with point patterns
            we wish to compute the structure factor of
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

        g2 = sample_xd_g2(coords, lv)  # returns ...
        eres.append(g2)

    eres = np.asarray(eres)
    ensemble_avg = np.mean(eres, axis=0)
    angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)

    if write_output:
        write_xd_data(ensemble_avg, "g2_res.txt", header="Isotropic pair correlation function")
        write_xd_data(
            angular_avg, "g2_bin.txt", header="Isotropic angular averaged pair correlation function"
        )

    return ensemble_avg, angular_avg


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    # CONFIG_DIR = sys.argv[1]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"
    CONFIG_FILE = "./initial_patterns/g2_res.txt"
    CONFIG_DIR = "./initial_patterns/"

    # res = testing(CONFIG_DIR, write_output=True)

    sys.path.append("/home/sd2402/n_point_statistics")
    from utils.readwrite import read_xd_data

    data = read_xd_data(CONFIG_FILE)
    bins, g2 = data.T

    res = fft_1d_g2(g2)
