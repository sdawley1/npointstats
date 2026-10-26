#!/bin/bash
"""
Script for sampling 2D Structure Factor and any related subroutines
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import os
import sys
import glob
import math
import functools
import itertools

import numpy as np

import jax
import jax.numpy as jnp


MAX_BLOCKS_PER_GRID = 65535


def gen_k_vecs(primitive: jnp.array) -> tuple:
    """
    Take d-dimensional primitive lattice vectors and
        return associate reciprocal lattice vectors
    Tranposition done to ensure reciprocal vectors are arranged
        in same fashion that primitive vectors are
    Args
        primitive = (d, d) ndarray of primitive lattice vectors w/ form
                [[x1, y2, z1, ...], [x2, y2, z2, ...], [x3, y3, z3,  ....], ...]
    Returns
        (d, d) ndarray of reciprocal lattice vectors
    """
    return (2 * jnp.pi * jnp.linalg.inv(primitive)).T


@functools.cache
def generate_wavevectors(n_waves: int, kn: tuple, kv: tuple) -> None:
    """
    Generates n_waves d-dimensional wave vectors given reciprocal lattice vectors
    Uses a cache so that sampleXDsf does not need to
        recompute wave vectors everytime it is called
    Inputs are tuples so that cache can work
    Args
        n_waves = integer number of wave vectors to generate
        kn = principle numbers [nx, ny, nz, ...] which correspond to the wave vectors 2pi*(nx/Lx, ny/Ly, nz/Lz, ...)
        kv = (d, d) tuple of tuples of reciprocal lattice vectors
    Returns
    """
    dimension = len(kv)
    wvs = np.zeros((n_waves, dimension))
    vector_indices = itertools.product(*map(range, kn))  # not even god herself is this good

    vidx = 0
    for ijk in vector_indices:
        # omit forward scattering
        if vidx == 0:
            vidx += 1
            continue

        wvs[vidx - 1] = jnp.dot(ijk, kv)
        vidx += 1

    # sort wave vectors by magnitude
    tmp = jnp.linalg.norm(wvs, axis=1)
    tmp = jnp.argsort(tmp)

    return wvs[tmp]


@jax.jit
def sample_2d_sf(sf: jnp.array, point_pattern: jnp.array, waves: jnp.array) -> None:
    """
    Compute 2D isotropic structure factor of given point pattern
    Args
        sf = (N, d+m) array containing [wave vector, structure factor] for all N points in point_pattern
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        waves = (M, d) array of wave vectors to use in computing structure factor
    """

    # metadata
    n_particles = point_pattern.shape[0]

    sinkr = 0
    coskr = 0
    for i in range(waves.shape[0]):
        k_vector = waves[i]
        knorm = jnp.hypot(k_vector[0], k_vector[1])

        for m in range(n_particles):
            pt = point_pattern[m]
            kdr = k_vector[0] * pt[0] + k_vector[1] * pt[1]
            coskr += math.cos(kdr)
            sinkr += math.sin(kdr)

        ntilde_squared = coskr**2 + sinkr**2

        # pylint: disable=E1121
        sf.at[i, 0].set(k_vector[0])
        sf.at[i, 1].set(k_vector[1])
        sf.at[i, 2].set(knorm)
        sf.at[i, 3].set(ntilde_squared / n_particles)

    return


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


def testing(idir: str, write_output: bool = False):
    """
    DOCSTRING
    Args
        idir = path to input directory containing files with point patterns we wish to compute the structure factor of
    """
    # pylint: disable=E0401
    # sys.path.append("/home/sd2402/n-point_statistics")

    # from utils.geom import angular_average
    # from utils.readwrite import read_cco, write_xd_data

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
        sfxv = sample_2d_sf(coords, lv)  # returns (W, d+2) ndarray
        eres.append(sfxv)

    # eres = np.asarray(eres)
    ensemble_avg = np.mean(eres, axis=0)
    # angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)

    # if write_output:
    #     write_xd_data(
    #         ensemble_avg, "sf_res.txt", header="Isotropic structure factor"
    #     )
    #     write_xd_data(
    #         angular_avg,
    #         "sf_bin.txt",
    #         header="Isotropic angular averaged structure factor",
    #     )

    # return ensemble_avg, angular_avg


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    # CONFIG_DIR = sys.argv[1]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    res = testing(CONFIG_DIR, write_output=False)
