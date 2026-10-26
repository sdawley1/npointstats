#!/bin/bash
"""
Script for sampling Structure Factor and related subroutines

Sam Dawley
12/2024
"""

import functools
import glob
import itertools
import os
from pathlib import Path
import sys
import time

import finufft
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ==================================================
# FILE I/O
# ==================================================


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


def write_xd_data(z: np.ndarray, ofile: str, delim: str = ",", header: str = None) -> bool:
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
            if header:
                hlines = header.split("\n")
                for hl in hlines:
                    f.write("# " + hl + "\n")

            for zz in z:
                s = delim.join(map(str, zz)) + "\n"
                f.write(s)

        print(f"Written to {ofile}...")
        return True

    except:
        print("Output unsuccessfully written")
        return False


# ==================================================
# STRUCTURE FACTOR
# ==================================================


def angular_average(
    Z: np.array, xidx: int = 0, yidx: int = -1, dk: float = 2 * np.pi
) -> np.array:
    """
    Parameters:
    -----------
        Z : np.ndarray, shape (N, M)
            output with data to be binned w.r.t. xidx and averaged over yidx
        xidx : int
            index within element of Z array for which we are binning
        yidx : int
            index within element of Z array for which we are averaging
        dk : float
            size of bins

    Returns:
    --------
        W : np.ndarray, shape (n_waves, 2)
    """
    X = Z[:, xidx]
    Y = Z[:, yidx]

    bins = np.arange(np.min(X), np.max(X), dk)
    bindices = np.digitize(X, bins)  # indices of bins
    unique, unique_indices, unique_counts = np.unique(
        bindices, return_index=True, return_counts=True
    )  # shapes = (N, 1) / (N, 1) / (M, 1) / (N, 1)

    W = np.zeros((unique.size, 2))
    for idx in range(unique.size - 1):
        ux1, ux2 = unique_indices[[idx, idx + 1]]  # indices of X, bindices
        ucnt = unique_counts[idx]

        ux = np.sum(X[ux1:ux2]) / ucnt
        uy = np.sum(Y[ux1:ux2]) / ucnt

        W[idx] = [ux, uy]

    return W[:-1]


def gen_k_vecs(primitive: np.ndarray) -> tuple:
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
    return (2 * np.pi * np.linalg.inv(primitive)).T


@functools.cache
def generate_wavevectors(n_waves: int, kn: tuple, kv: tuple) -> np.ndarray:
    """
    Generates n_waves d-dimensional wave vectors given reciprocal lattice vectors
    Uses a cache so that sampleXDsf does not need to
        recompute wave vectors everytime it is called
    Inputs are tuples so that cache can work
    Parameters:
    -----------
        n_waves : int
            integer number of wave vectors to generate
        kn : tuple
            principle numbers [nx, ny, nz, ...] which correspond to the wave vectors 2pi*(nx/Lx, ny/Ly, nz/Lz, ...)
        kv : np.ndarray
            (d, d) tuple of tuples of reciprocal lattice vectors
    Returns:
    --------
        wvs : np.ndarray
            (n_waves, d) ndarray of wave vectors
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

        wvs[vidx - 1] = np.dot(ijk, kv)
        vidx += 1

    # sort wave vectors by magnitude
    tmp = np.linalg.norm(wvs, axis=1)
    tmp = np.argsort(tmp)

    # return wvs[tmp]
    return wvs


def sample_xd_sf(
    point_pattern: np.ndarray,
    lv: np.ndarray,
    kn: tuple = (1_000_000, 150, 150),  # 1D
    # kn: tuple = (150, 150, 150) # 3D
) -> None:
    """
    Sample d-dimensional isotropic structure factor S(k)

    Parameters:
    -----------
        point_pattern : ndarray, shape (N, d)
            d-dimensional coordinates of all N points
        lv : ndarray, shape (d, d)
            primitive lattice vectors
        kn : ndarray, shape (d,)
            maximum integer multiple of wave vector to consider along each dimension

    Returns:
    --------
        sfxv : np.ndarray
            (W, d+2) ndarray with elements [k_{x1}, ..., k_{xd}, |k|, S(k)] for all W wave vectors generated
    """

    point_pattern = np.asarray(point_pattern)
    n_particles, dimension = point_pattern.shape

    # ----- generate wave vectors -----

    match dimension:
        case 1:
            kn = (100_000, )
        case 2:
            kn = (500, 500)
        case 3:
            kn = (25, 25, 25)

    kn = kn[:dimension]
    n_waves = int(np.prod(kn))
    # build integer mode indices in unshifted order (0,1,...,N/2-1,-N/2,...,-1)
    mode_axes = [(np.fft.fftfreq(n) * n).astype(int) for n in kn]
    wave_indices = np.asarray(list(itertools.product(*mode_axes)))

    rv = gen_k_vecs(lv)
    wave_vectors = wave_indices @ rv

    sfxv = np.zeros((n_waves, dimension + 2), dtype=np.float64)
    sfxv[:, :dimension] = wave_vectors
    sfxv[:, -2] = np.linalg.norm(wave_vectors, axis=1)

    # for use with FINUFFT, points must lie on interval [-pi, pi)
    for k in range(dimension):
        point_pattern[:, k] /= lv[k, k]
    point_pattern = np.add(np.multiply(point_pattern, 2 * np.pi), -np.pi)

    # ----- compute structure factor -----

    match dimension:
        case 1:
            ntilde = finufft.nufft1d1(
                np.ascontiguousarray(point_pattern)[:, 0],
                np.ones(n_particles).astype(complex),
                n_modes=kn,
                eps=1e-12,
            )

        case 2:
            ntilde = finufft.nufft2d1(
                np.ascontiguousarray(point_pattern[:, 0]),
                np.ascontiguousarray(point_pattern[:, 1]),
                np.ones(n_particles).astype(complex),
                n_modes=kn,
                eps=1e-12,
            )

        case 3:
            ntilde = finufft.nufft3d1(
                np.ascontiguousarray(point_pattern[:, 0]),
                np.ascontiguousarray(point_pattern[:, 1]),
                np.ascontiguousarray(point_pattern[:, 2]),
                np.ones(n_particles).astype(complex),
                n_modes=kn,
                eps=1e-12,
            )

    # move zero-frequency to index 0 along all axes for plotting
    ntilde = np.fft.ifftshift(ntilde)
    sf = np.abs(ntilde) ** 2 / n_particles
    sfxv[:, -1] = sf.flatten()

    sindices = np.argsort(sfxv[:, -2])
    sfxv = sfxv[sindices][1:]  # ignore forward scattering

    del point_pattern, n_particles, dimension, rv, kn, n_waves, ntilde, sf
    return sfxv


# ==================================================
# COLLECTIVE COORDINATE IMPLEMENTATION
# ==================================================


@jax.jit
def _compute_sf_single_k(k_vector: jnp.ndarray, point_pattern: jnp.ndarray) -> jnp.ndarray:
    """
    Compute structure factor for a single wave vector using collective coordinates.
    
    S(k) = |ρ̃(k)|² / N  where  ρ̃(k) = Σⱼ exp(i k · rⱼ)
    
    Parameters:
    -----------
        k_vector : jnp.ndarray, shape (d,)
            Single wave vector
        point_pattern : jnp.ndarray, shape (N, d)
            Particle positions
            
    Returns:
    --------
        jnp.ndarray, shape (d+2,)
            [k_x, k_y, ..., |k|, S(k)]
    """
    n_particles = point_pattern.shape[0]
    
    kdr = jnp.dot(point_pattern, k_vector)
    
    coskr = jnp.sum(jnp.cos(kdr))
    sinkr = jnp.sum(jnp.sin(kdr))
    
    ntilde_squared = coskr ** 2 + sinkr ** 2
    sk = ntilde_squared / n_particles
    
    knorm = jnp.linalg.norm(k_vector)
    
    return jnp.concatenate([k_vector, jnp.array([knorm, sk])])


@jax.jit
def _compute_sf_all_k(waves: jnp.ndarray, point_pattern: jnp.ndarray) -> jnp.ndarray:
    """
    Compute structure factor for all wave vectors (vectorized).
    
    Parameters:
    -----------
        waves : jnp.ndarray, shape (M, d)
            Array of wave vectors
        point_pattern : jnp.ndarray, shape (N, d)
            Particle positions
            
    Returns:
    --------
        jnp.ndarray, shape (M, d+2)
            [k_x, k_y, ..., |k|, S(k)] for each wave vector
    """
    return jax.vmap(lambda k: _compute_sf_single_k(k, point_pattern))(waves)


def sample_xd_sf_collective(
    point_pattern: np.ndarray,
    lv: np.ndarray,
    kn: tuple = None,
) -> np.ndarray:
    """
    Sample d-dimensional isotropic structure factor S(k) via collective coordinates.
    
    Uses JAX for JIT compilation and vectorization. Computes:
        S(k) = |ρ̃(k)|² / N  where  ρ̃(k) = Σⱼ exp(i k · rⱼ)
    
    This mirrors the GPU implementation in g_structure_factor.py but runs on CPU.

    Parameters:
    -----------
        point_pattern : ndarray, shape (N, d)
            d-dimensional coordinates of all N points
        lv : ndarray, shape (d, d)
            primitive lattice vectors
        kn : tuple, optional
            maximum integer multiple of wave vector to consider along each dimension
            If None, uses defaults based on dimension

    Returns:
    --------
        sfxv : np.ndarray
            (W, d+2) ndarray with elements [k_{x1}, ..., k_{xd}, |k|, S(k)] for all W wave vectors generated
    """
    ts = time.time()
    
    point_pattern = np.asarray(point_pattern, dtype=np.float64)
    lv = np.asarray(lv, dtype=np.float64)
    _, dimension = point_pattern.shape

    rv = gen_k_vecs(lv)
    rvt = tuple([tuple(a) for a in rv])

    if kn is None:
        match dimension:
            case 1:
                kn = (150,)
            case 2:
                kn = (250, 250)
            case 3:
                kn = (15, 15, 15)

    n_waves = int(np.prod(kn)) - 1
    waves = generate_wavevectors(n_waves, kn, rvt)
    
    # sort wave vectors by magnitude for consistency with GPU version
    sort_idx = np.argsort(np.linalg.norm(waves, axis=1))
    waves = waves[sort_idx]
    
    print(f"{waves.shape=}", flush=True)

    g_coords = jnp.asarray(point_pattern)
    g_waves = jnp.asarray(waves)

    sfxv = _compute_sf_all_k(g_waves, g_coords)
    sfxv = np.asarray(sfxv)

    te = time.time()
    print(f"elapsed time = {(te-ts):0.4f}")

    return sfxv


# ==================================================
# TESTING
# ==================================================


def testing(idir: str, fpx: str, write_output: bool = False):
    """
    DOCSTRING
    Args
        idir = path to input directory containing files with point patterns we wish to compute the structure factor of
    """
    idir = Path(idir)

    # determine if we are using a directory of inputs (disordered system)
    # or just a single input (ordered system)
    if os.path.isdir(idir):
        path = idir.joinpath(fpx)
        configs = glob.glob(str(path))

    else:
        sys.exit(1)

    # metadata
    n_members = len(configs)
    eres = []  # ensemble averaging

    for midx in range(n_members):
        config = configs[midx]
        _, _, lv, coords = read_cco(config)

        sfxv = sample_xd_sf(coords, lv)  # returns (W, d+2) ndarray
        eres.append(sfxv)

    # eres = np.asarray(eres)
    ensemble_avg = np.mean(eres, axis=0)
    angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)

    if write_output:
        write_xd_data(ensemble_avg, "sf_res.txt", header="Isotropic structure factor")
        write_xd_data(
            angular_avg,
            "sf_bin.txt",
            header="Isotropic angular averaged structure factor",
        )

    return ensemble_avg


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    CONFIG_DIR = sys.argv[1]
    FILE_PFX = sys.argv[2]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    res, aa_res = testing(CONFIG_DIR, FILE_PFX, write_output=False)
