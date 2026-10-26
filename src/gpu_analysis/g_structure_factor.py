#!/bin/bash
"""
Script for sampling 2D Structure Factor and any related subroutines
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import ctypes
import functools
import glob
import itertools
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import numba as nb
from numba import cuda

SPHERES_C = ctypes.cdll.LoadLibrary("/home/sd2402/stats/utils/lib_spheres.so")

sbj = SPHERES_C.besselJ
sbj.argtypes = [ctypes.c_double, ctypes.c_double]
sbj.restype = ctypes.c_double


MAX_BLOCKS_PER_GRID = 65_535


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


# ==================================================
# POTPOURRI
# ==================================================


@nb.njit(fastmath=True)
def besselJ_c(v: float, x: float) -> float:
    """Compute bessel function using C backend"""
    return sbj(v, x)


besselJ_cwrap = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(besselJ_c)


@nb.njit(fastmath=True)
def besselJ(v: float, x: float) -> float:
    """Return vth-order Bessel Function of first kind at x with nterms in series expansion"""
    return besselJ_cwrap(v, x)


@nb.njit(fastmath=True)
def fsiv(k: float, R: float, d: int) -> float:
    """
    Return Fourier transform of scaled intersection volume alpha_2(k;R)
    cf. Torquato's 'Hyperuniform States of Matter', pg. 15 Eq. 49
        for how to check that this func is working (I have and it is)
    Args
        k = floating point magnitude of wave vector connecting sphere centroids
        R = floating point radius of spheres (real space)
        d = dimensionality
    Returns
        floating point scaled intersection volume
    """
    return (2 * np.sqrt(np.pi) / k) ** d * math.gamma(1 + d / 2) * besselJ(d / 2, k * R) ** 2


@nb.njit(fastmath=True)
def trapz_integrate(X: np.array, Y: np.array):
    """
    Numerical integration of Y over X
    Follows numerical recipe for algorithm taken from
        https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter21.03-Trapezoid-Rule.html
    Mine works and is numba-friendly but NumPy is more accurate (shocking)
    Args
        X, Y = same-size arrays containing independent, dependent data to integrate over
        mode = string of which method to use for numerical integration.
                If not supplied, default to np.trapz
    Returns
        I = floating point numerical approximation to integral f over [a, b]
    """
    # Parameters specfic to this function call
    func = Y
    domain = X
    N = domain.size

    # Assumes domain is well-ordered
    harray = [domain[i + 1] - domain[i] for i in range(domain.size)]
    harray.insert(0, 0)

    # # TRAPEZOID RULE
    # I = harray[0]*func[0]/2  + harray[-1]*func[-1]/2

    # for i in range(1, N-1):
    #     I += harray[i]*func[i]

    # return I

    # SIMPSON'S RULE
    I = harray[0] * func[0] / 3 + harray[-1] * func[-1] / 3

    for i in range(2, N - 2, 2):
        I += 2 * harray[i] * func[i] / 3

    for i in range(1, N - 1, 2):
        I += 4 * harray[i] * func[i] / 3

    return I


# ==================================================
# STRUCTURE FACTOR
# ==================================================


def gen_k_vecs(primitive: np.array) -> tuple:
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
def generate_wavevectors(n_waves: int, kn: tuple, kv: tuple) -> np.array:
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
        kv : np.array
            (d, d) tuple of tuples of reciprocal lattice vectors
    Returns:
    --------
        wvs : np.array
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

    return wvs[tmp]


def sample_xd_sf(point_pattern: np.array, lv: np.array, tpb: int = 128) -> None:
    """
    Sample d-dimensional isotropic structure factor S(k)

    Parameters:
    -----------
        point_pattern : ndarray, shape (N, d)
            d-dimensional coordinates of all N points
        lv : ndarray, shape (d, d)
            primitive lattice vectors
        tpb : int
            threads per block to use for GPU

    Returns:
    --------
        sfxv : np.array
            (W, d+2) ndarray with elements [k_{x1}, ..., k_{xd}, |k|, S(k)] for all W wave vectors generated
    """
    ts = time.time()

    # derived metadata
    point_pattern = np.asarray(point_pattern)
    _, dimension = point_pattern.shape

    rv = gen_k_vecs(lv)
    rvt = tuple([tuple(a) for a in rv])

    match dimension:
        case 1:
            kn = (150,)
        case 2:
            kn = (50, 50,)
        case 3:
            kn = (15, 15, 15,)

    n_waves = int(np.prod(kn)) - 1  # omit forward scattering
    waves = generate_wavevectors(n_waves, kn, rvt)

    sfxv = np.zeros((n_waves, dimension + 2))

    bpg = int(np.ceil(sfxv.shape[0] / tpb))
    if bpg * tpb > MAX_BLOCKS_PER_GRID:
        raise ValueError(
            f"Too many blocks per grid for GPU to handle: {(bpg * tpb // 1):d} > {MAX_BLOCKS_PER_GRID:d}"
        )

    g_coords = cuda.to_device(point_pattern)
    g_waves = cuda.to_device(waves)
    g_sfxv = cuda.to_device(sfxv)

    print(f"{waves.shape=}", flush=True)

    if dimension == 1:
        sample_1d_sf[tpb, bpg](g_sfxv, g_coords, g_waves)

    elif dimension == 2:
        sample_2d_sf[tpb, bpg](g_sfxv, g_coords, g_waves)

    elif dimension == 3:
        sample_3d_sf[tpb, bpg](g_sfxv, g_coords, g_waves)

    sfxv = g_sfxv.copy_to_host()

    te = time.time()
    print(f"elapsed time = {(te-ts):0.4f}")

    return sfxv


@cuda.jit(fastmath=True)
def sample_1d_sf(sf: np.array, point_pattern: np.array, waves: np.array) -> None:
    """
    Compute 1D structure factor of given point pattern
    Args
        sf = (N, d+m) array containing [wave vector, structure factor] for all N points in point_pattern
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        waves = (M, d) array of wave vectors to use in computing structure factor
    """
    i = cuda.grid(ndim=1)
    if i >= sf.shape[0]:
        return

    # metadata
    n_particles = point_pattern.shape[0]
    k_vector = waves[i]

    sinkr = 0
    coskr = 0
    for m in range(n_particles):
        pt = point_pattern[m]
        kdr = k_vector[0] * pt[0]
        coskr += math.cos(kdr)
        sinkr += math.sin(kdr)

    ntilde = coskr**2 + sinkr**2
    knorm = abs(k_vector[0])

    # pylint: disable=E1121
    cuda.atomic.add(sf, (i, 0), k_vector[0])
    cuda.atomic.add(sf, (i, 1), knorm)
    cuda.atomic.add(sf, (i, 2), ntilde / n_particles)

    return


@cuda.jit(fastmath=True)
def sample_2d_sf(sf: np.array, point_pattern: np.array, waves: np.array) -> None:
    """
    Compute 2D isotropic structure factor of given point pattern
    Args
        sf = (N, d+m) array containing [wave vector, structure factor] for all N points in point_pattern
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        waves = (M, d) array of wave vectors to use in computing structure factor
    """
    i = cuda.grid(ndim=1)
    if i >= sf.shape[0]:
        return

    # metadata
    n_particles = point_pattern.shape[0]
    k_vector = waves[i]
    knorm = np.hypot(k_vector[0], k_vector[1])

    sinkr = 0
    coskr = 0
    for m in range(n_particles):
        pt = point_pattern[m]
        kdr = k_vector[0] * pt[0] + k_vector[1] * pt[1]
        coskr += math.cos(kdr)
        sinkr += math.sin(kdr)

    ntilde = coskr**2 + sinkr**2

    # pylint: disable=E1121
    cuda.atomic.add(sf, (i, 0), k_vector[0])
    cuda.atomic.add(sf, (i, 1), k_vector[1])
    cuda.atomic.add(sf, (i, 2), knorm)
    cuda.atomic.add(sf, (i, 3), ntilde / n_particles)

    return


@cuda.jit(fastmath=True)
def sample_3d_sf(sf: np.array, point_pattern: np.array, waves: np.array) -> None:
    """
    Compute 3D isotropic structure factor of given point pattern
    Args
        sf = (N, d+m) array containing [wave vector, structure factor] for all N points in point_pattern
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        waves = (M, d) array of wave vectors to use in computing structure factor
    """
    i = cuda.grid(ndim=1)
    if i >= sf.shape[0]:
        return

    # metadata
    n_particles = point_pattern.shape[0]
    k_vector = waves[i]
    knorm = math.sqrt(k_vector[0] ** 2 + k_vector[1] ** 2 + k_vector[2] ** 2)

    sinkr = 0
    coskr = 0
    for m in range(n_particles):
        pt = point_pattern[m]
        kdr = k_vector[0] * pt[0] + k_vector[1] * pt[1] + k_vector[2] * pt[2]
        coskr += math.cos(kdr)
        sinkr += math.sin(kdr)

    ntilde = coskr**2 + sinkr**2

    # pylint: disable=E1121
    cuda.atomic.add(sf, (i, 0), k_vector[0])
    cuda.atomic.add(sf, (i, 1), k_vector[1])
    cuda.atomic.add(sf, (i, 2), k_vector[2])
    cuda.atomic.add(sf, (i, 3), knorm)
    cuda.atomic.add(sf, (i, 4), ntilde / n_particles)

    return


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
    Returns
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
    # angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)

    if write_output:
        write_xd_data(ensemble_avg, "sf_res.txt", header="Isotropic structure factor")
        # write_xd_data(
        #     angular_avg,
        #     "sf_bin.txt",
        #     header="Isotropic angular averaged structure factor",
        # )

    return ensemble_avg


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.4e}"})

    # SPECIFY INPUT DIRECTORY HERE
    CONFIG_DIR = sys.argv[1]
    FILE_PFX = sys.argv[2]
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/patterns/"
    # CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/"

    res, aa_res = testing(CONFIG_DIR, FILE_PFX, write_output=False)
