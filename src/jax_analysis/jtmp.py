#!/bin/bash
"""
Script for sampling 2D Structure Factor and any related subroutines
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import glob
import math
import functools
import itertools

import numpy as np
from jax import jit
import jax.numpy as jnp


MAX_BLOCKS_PER_GRID = 65535


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


@jit
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


# @functools.cache
def generate_wavevectors(n_waves: int, kn: jnp.array, kv: jnp.array) -> None:
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
    wvs = jnp.zeros((n_waves, dimension))
    vector_indices = itertools.product(*map(range, kn))  # not even god herself is this good

    vidx = 0
    for ijk in vector_indices:
        hkl = jnp.asarray(ijk)

        # omit forward scattering
        if vidx == 0:
            vidx += 1
            continue

        wvs = wvs.at[vidx - 1].set(jnp.dot(hkl, kv))
        vidx += 1

    # sort wave vectors by magnitude
    tmp = jnp.linalg.norm(wvs, axis=1)
    tmp = jnp.argsort(tmp)

    return wvs[tmp]


@jit
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
            coskr += jnp.cos(kdr)
            sinkr += jnp.sin(kdr)

        ntilde_squared = coskr**2 + sinkr**2

        # pylint: disable=E1121
        sf.at[i, 0].set(k_vector[0])
        sf.at[i, 1].set(k_vector[1])
        sf.at[i, 2].set(knorm)
        sf.at[i, 3].set(ntilde_squared / n_particles)

    return


def main(config_dir: str) -> None:
    """
    DOCSTRING
    """
    # metadata
    nx, ny, nz = 128, 128, 1
    configs = glob.glob("*__*", root_dir=config_dir)

    n_members = len(configs)
    # eres = []  # ensemble averaging

    for midx in range(n_members):
        print(f"Member {midx}")

        config = configs[midx]
        dimension, _, lv, coords = read_cco(config_dir + config)

        rv = gen_k_vecs(lv)
        kn = jnp.array([nx, ny, nz])[:dimension]

        n_waves = int(jnp.prod(kn)) - 1  # omit forward scattering
        waves = generate_wavevectors(n_waves, kn, rv)

        sfxv = jnp.zeros((n_waves, dimension + 2))
        sample_2d_sf(sfxv, coords, waves)  # returns (W, d+2) ndarray


if __name__ == "__main__":

    CONFIGURATIONS = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    main(CONFIGURATIONS)
