#!/bin/bash
"""
Script for sampling local number variance
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import os
import sys
import glob
import time
import math

import freud
import numpy as np
import numba as nb


def read_cco(ifile: str) -> tuple:
    """
    Read data from CCO, taking advantage of fixed file setup
    Returns
        d               = system dimensionality
        num_particles   = number of particles
        lattice_vectors = primitive lattice vectors of 3D system
        coordinates     = positions of all individual spheres
    """
    lattice_vectors = []
    coordinates = []

    with open(ifile, "r", encoding="utf-8") as f:
        lines = f.readlines()
        d = int(lines[0].strip("\n"))

        # d counts number of vectors, -1 for dimension in the first line
        n_particles = len(lines) - (d + 1)

        for i in range(1, d + 1):  # Get all basis vectors
            basis_vector = lines[i].strip("\n").split()
            lattice_vectors.append([float(a) for a in basis_vector[:d]])

        for j in range(d + 1, len(lines)):  # Get all coordinates
            coord = lines[j].strip("\n").split()
            coordinates.append([float(a) for a in coord[:d]])

    return d, n_particles, np.array(lattice_vectors), np.array(coordinates)


def write_xd_data(z: np.array, ofile: str, delim: str = ",", header: str = None) -> bool:
    """
    Write data contained in Z to comma-separated text file `ofile`
    Args
        z = (M, N) ndarray of data to write to output
        ofile = name of output file
    Returns
        True (False) if z written to ofile (un)successfully
    """
    try:
        with open(ofile, "w", encoding="utf-8") as f:
            # write header
            if header:
                hlines = header.split("\n")
                for hl in hlines:
                    f.write("# " + hl + "\n")

            # write data
            for zz in z:
                s = delim.join(map(str, zz)) + "\n"
                f.write(s)

        print(f"Written to {ofile}...")
        return True

    except:
        print("Output unsuccessfully written")
        return False


@nb.njit(fastmath=True)
def sphere_volume(R: float, d: int) -> float:
    """Compute volume of d-dimensional sphere of radius R"""
    return np.pi ** (d / 2) * R**d / math.gamma(1 + d / 2)


def sample_xd_nv(point_pattern: np.array, lv: np.array, delta_r: float = 2.5e-2) -> None:
    """
    Sample d-dimensional local number variance
    Args
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        lv = (d, d) ndarray of primitive lattice vectors
        delta_r = spacing between successive radii for sampling
    Returns
        nv = (W, 2) ndarray with elements [R, sigma^2(R)] for all W radii we consider
    """
    ts = time.time()

    # metadata
    M = 500

    # derived metadata
    _, d = point_pattern.shape

    # freud shenanigans
    lengths = list(map(np.linalg.norm, lv))
    f_box = freud.box.Box(*lengths)
    f_pts = np.zeros((point_pattern.shape[0], 3))

    for lidx, length in enumerate(lengths):
        f_pts[:, lidx] = point_pattern[:, lidx] - length / 2

    aq = freud.locality.LinkCell(f_box, f_pts)
    query_points = np.min(lengths) * np.random.rand(M, 3) - np.min(lengths) / 2

    bins = np.arange(delta_r, np.max(lengths) / 4, delta_r)
    nv = np.zeros((bins.size, 2))
    n_pts_in_centroids = np.zeros((bins.size, M))

    for bidx in range(bins.size):
        radius = bins[bidx]

        q_res = aq.query(query_points, {"mode": "ball", "r_max": radius, "exclude_ii": True})
        for bond in q_res:
            q_idx, _, _ = bond
            n_pts_in_centroids[bidx, q_idx] += 1

    te = time.time()
    print(f"elapsed time = {(te-ts):0.4f}")

    nv[:, 0] = bins
    nv[:, 1] = np.var(n_pts_in_centroids, axis=1) / sphere_volume(bins, d)

    return nv


def testing(idir: str, write_output: bool = False):
    """
    DOCSTRING
    Args
        idir = path to input directory containing files with point patterns we wish to compute the structure factor of
    """

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

        nv = sample_xd_nv(coords, lv)  # returns ...
        eres.append(nv)

    eres = np.asarray(eres)
    ensemble_avg = np.mean(eres, axis=0)

    if write_output:
        write_xd_data(ensemble_avg, "nv_res.txt", header="Isotropic local number variance")

    return ensemble_avg


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    CONFIG_DIR = sys.argv[1]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    res = testing(CONFIG_DIR, write_output=True)
