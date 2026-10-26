# %%

#!/bin/bash
"""
Script for sampling n-pt correlation functions
Utilizes CUDA toolkit to acceralate computations w/ GPUs

Sam Dawley
12/2024
"""

import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"

import jax
import jax.numpy as jnp

# import matplotlib.pyplot as plt

# pylint: disable=E0401
sys.path.append("/home/sd2402/n_point_statistics")

# from utils.geom import jitted_nearest_image, nearest_image, parallelipiped_volume
from utils.readwrite import read_cco, write_xd_data


# ==================================================
# MISCELLANEOUS
# ==================================================

def parallelipiped_volume(basis_vectors: np.array) -> float:
    """Return numba-friendly volume of d-dimensional parallelipiped with known basis vectors"""
    return np.abs(np.linalg.det(basis_vectors))

def sphere_volume(R: float, d: int) -> float:
    """Compute volume of d-dimensional sphere of radius R"""
    return np.pi ** (d / 2) * R**d / math.gamma(1 + d / 2)

def shell_volume(R: float, d: int, dr: float) -> float:
    """
    Compute volume of d-dimensional shell (annulus) of inner radius R and outer radius R+dr
    Args
        R = floating point inner radius of annulus
        d = integer dimensionality
        dr = floating point width of annulus
    Returns
        volume of d-dimensional annulus
    """
    return sphere_volume(R, d) * ((R + dr / 2) ** d - (R - dr / 2) ** d) / (R**d)

def tuple_to_contiguous_index(pair: tuple, n_rows: int) -> int:
    r"""
    Convert $(m,n)\in\mathbb{Z}^N\times\mathbb{Z}^N$ to
        integer $k\in\mathbb{Z}^N$ w/ injective mapping
    Useful for mapping indices of 2D ndarray to contiguous 1D array
    Args
        pair = (m, n) tuple of 2D index
        n_rows = integer number of rows in 2D array
    """
    i, j = pair
    return int(i * (n_rows - 1) - i * (i + 1) / 2 + j - 1)


# def nearest_image(primitive: np.array, p1: np.array, p2: np.array = None) -> np.array:
#     """
#     Compute d-dimensional Euclidean distance between points p1, p2 using nearest-image convention
#     Assumes center of primitive lattice is at CORNER of parallelogram defined by primitive
#     Args
#         p1 = coordinates of tagged point
#         p2 = coordinates of other point
#         primitive = primitive lattice vectors
#     """
#     dimension = p1.size
#     primitive_norms = map(np.linalg.norm, primitive)

#     # Nearest-image position of a single particle
#     if p2 is None:
#         for i in range(dimension):
#             k = next(primitive_norms)

#             if p1[i] >= k:
#                 p1[i] -= k

#             elif p1[i] <= 0:
#                 p1[i] += k

#         return p1

#     # Nearest-image distance between two particles
#     dr = p1 - p2
#     for i in range(dimension):
#         k = next(primitive_norms)

#         if dr[i] >= k / 2:
#             dr[i] -= k

#         elif dr[i] <= -k / 2:
#             dr[i] += k

#     return np.asarray(dr)


# ==================================================
# 2-PT CORRELATION FUNCTION SAMPLING
# ==================================================


def dist_matrix_1d(
    arr1: jnp.ndarray,
    arr2: jnp.ndarray,
    primitive: Tuple[Tuple[float, float], float] # jax aliasing or sumthin...
) -> jnp.ndarray:
    """
    Compute pairwise distances between two 1D coordinate arrays with periodic boundary conditions.
    Uses nearest image convention with box length taken from the primitive lattice vector.
    Args
        arr1: (M, 1) array of coordinates
        arr2: (N, 1) array of coordinates
        primitive: (1, 1) ndarray of primitive lattice vectors
    Returns
        (M, N) array of pairwise distances
    """

    cx = float(np.abs(primitive[0][0]))

    dx = arr1[:, 0][:, None] - arr2[:, 0][None, :]

    half_cx = cx / 2.0

    dx = jnp.where(dx >= half_cx, dx - cx, dx)
    dx = jnp.where(dx <= -half_cx, dx + cx, dx)

    return jnp.abs(dx)


def dist_matrix_2d(
    arr1: jnp.ndarray,
    arr2: jnp.ndarray,
    primitive: Tuple[Tuple[float, float], float] # jax aliasing or sumthin...
) -> jnp.ndarray:
    """
    Compute pairwise distances between two 2D coordinate arrays with periodic boundary conditions.
    Uses the nearest image convention with box lengths taken from the primitive lattice vectors.
    Args
        arr1: (M, 2) array of coordinates
        arr2: (N, 2) array of coordinates
        primitive: (2, 2) ndarray of primitive lattice vectors
    Returns
        (M, N) array of pairwise distances
    """

    cx = float(np.hypot(primitive[0][0], primitive[0][1]))
    cy = float(np.hypot(primitive[1][0], primitive[1][1]))

    dx = arr1[:, 0][:, None] - arr2[:, 0][None, :]
    dy = arr1[:, 1][:, None] - arr2[:, 1][None, :]

    half_cx = cx / 2.0
    half_cy = cy / 2.0

    dx = jnp.where(dx >= half_cx, dx - cx, dx)
    dx = jnp.where(dx <= -half_cx, dx + cx, dx)
    dy = jnp.where(dy >= half_cy, dy - cy, dy)
    dy = jnp.where(dy <= -half_cy, dy + cy, dy)

    return jnp.hypot(dx, dy)


def dist_matrix_3d(
    arr1: jnp.ndarray,
    arr2: jnp.ndarray,
    primitive: Tuple[Tuple[float, float], float] # jax aliasing or sumthin...
) -> jnp.ndarray:
    """
    Compute pairwise distances between two 3D coordinate arrays with periodic boundary conditions.
    Uses nearest image convention with box lengths taken from the primitive lattice vectors.
    Args
        arr1: (M, 3) array of coordinates
        arr2: (N, 3) array of coordinates
        primitive: (3, 3) ndarray of primitive lattice vectors
    Returns
        (M, N) array of pairwise distances
    """

    cx = float(np.linalg.norm(primitive[0]))
    cy = float(np.linalg.norm(primitive[1]))
    cz = float(np.linalg.norm(primitive[2]))

    dx = arr1[:, 0][:, None] - arr2[:, 0][None, :]
    dy = arr1[:, 1][:, None] - arr2[:, 1][None, :]
    dz = arr1[:, 2][:, None] - arr2[:, 2][None, :]

    half_cx = cx / 2.0
    half_cy = cy / 2.0
    half_cz = cz / 2.0

    dx = jnp.where(dx >= half_cx, dx - cx, dx)
    dx = jnp.where(dx <= -half_cx, dx + cx, dx)
    dy = jnp.where(dy >= half_cy, dy - cy, dy)
    dy = jnp.where(dy <= -half_cy, dy + cy, dy)
    dz = jnp.where(dz >= half_cz, dz - cz, dz)
    dz = jnp.where(dz <= -half_cz, dz + cz, dz)

    return jnp.sqrt(dx * dx + dy * dy + dz * dz)


dist_matrix_1d = jax.jit(dist_matrix_1d, static_argnums=(2,))
dist_matrix_2d = jax.jit(dist_matrix_2d, static_argnums=(2,))
dist_matrix_3d = jax.jit(dist_matrix_3d, static_argnums=(2,))


def sample_xd_g2(
    point_pattern: np.ndarray,
    lv: np.ndarray,
    delta_r: Optional[float] = 5e-2,
    max_l: Optional[float] = None,
) -> np.ndarray:
    """
    DOCSTRING
    Args
        point_pattern = (N, d) array of d-dimensional coordinates of all N points
        lv = (d, d) ndarray of primitive lattice vectors
        delta_r = spacing between successive radii for binning result
    Returns
        g2_hist = (M, 2) ndarray of the two point correlation function
    """
    # derived metadata
    n_particles, dimension = point_pattern.shape
    vol = parallelipiped_volume(lv)
    rho = n_particles / vol

    delta_r /= vol ** (1 / dimension)

    if max_l is None:
        max_l = np.max(lv) / 2

    bins = np.arange(delta_r, np.max(lv) / 2, delta_r)
    rdiffs = np.ediff1d(bins)

    g2 = np.zeros((bins.size - 1, 2))

    point_pattern = jnp.asarray(point_pattern)
    lv = tuple([tuple(a) for a in lv])

    match dimension:
        case 1:
            radii = dist_matrix_1d(point_pattern, point_pattern, lv)
        case 2:
            radii = dist_matrix_2d(point_pattern, point_pattern, lv)
        case 3:
            radii = dist_matrix_3d(point_pattern, point_pattern, lv)

    hist, _ = np.histogram(radii.ravel(), bins)

    g2[:, 0] = bins[1:]
    g2[:, 1] = np.multiply(
        hist, 1 / shell_volume(bins[1:], dimension, rdiffs) / rho / n_particles
    )  # fixed in 2D

    return g2


# ==================================================
# 3-PT CORRELATION FUNCTION SAMPLING
# ==================================================


# def original_sample_2d_g3(
#     point_pattern: np.array, lv: np.array, delta_r: float = 5e-1, delta_theta: float = np.pi / 20
# ) -> np.array:
#     """
#     Brute force sampling of three-point correlation function g_3(r_12, r_13, theta)
#     Assumes box is anchored at (0, 0) with side lengths (Lx, Ly)
#     Args
#         point_pattern = (N, d) array of d-dimensional coordinates of all N points
#         lv = (d, d) ndarray of primitive lattice vectors
#         delta_r = spacing between successive radii for binning result
#         delta_theta = spacing between successive angles for binning result
#     Returns
#         g3_hist = (T, M, N) ndarray of the three point correlation function
#             axis = 0 (T): indexes the angle of the triangle
#             axis = 1 (M): indexes first side length of the triangle
#             axis = 2 (N): indexes second side length of the triangle
#             (T, M, N) depend on choice of (delta_theta, delta_r, delta_r) and size of simulation box
#     """
#     ts = time.time()

#     # derived metadata
#     n_particles, _ = point_pattern.shape
#     vol = parallelipiped_volume(lv)
#     rho = n_particles / vol

#     angle_bins = np.arange(delta_theta, np.pi + delta_theta, delta_theta)
#     radii_bins = np.arange(delta_r, np.max(lv) / 2, delta_r)

#     fixed_angle_counts = [[] for _ in range(angle_bins.size)]

#     g3_hist = np.zeros(
#         (angle_bins.size, radii_bins.size - 1, radii_bins.size - 1), dtype=np.float64
#     )
#     print(f"g3 size = {g3_hist.nbytes / 10**9:0.2e} Gb")

#     # massive loop for getting angle between triplets of particles
#     # don't consider (i > n_particles - 2) because a triplet cannot be formed in this case
#     # for i in range(n_particles - 2):
#     for i in range(2):
#         p1 = point_pattern[i]

#         large_dimension = n_particles - i - 1
#         small_dimension = n_particles - i - 2

#         unbinned_angles = np.zeros(large_dimension * small_dimension // 2, dtype=np.float64)
#         unbinned_sides = np.zeros((large_dimension * small_dimension // 2, 2), dtype=np.float64)

#         for j in range(i + 1, n_particles):
#             p2 = point_pattern[j]

#             v12 = nearest_image(lv, p2, p1)
#             v12_norm = np.linalg.norm(v12)  # distance between points (i, j)

#             for k in range(j + 1, n_particles):
#                 p3 = point_pattern[k]

#                 v13 = nearest_image(lv, p3, p1)
#                 v13_norm = np.linalg.norm(v13)  # distance between points (i, k)

#                 # angle between vectors connecting (i, j) and (i, k)
#                 # checks edge case where |varphi| slightly greater than \pi
#                 varphi = np.nanmin([3.1415, np.arccos(np.dot(v12, v13) / v12_norm / v13_norm)])

#                 jk = tuple_to_contiguous_index((j - i - 1, k - i - 1), n_particles - i - 1)

#                 unbinned_angles[jk] = varphi
#                 unbinned_sides[jk] = [v12_norm, v13_norm]

#         angle_bindices = np.digitize(unbinned_angles, angle_bins)

#         for ua_idx in range(unbinned_angles.size):
#             angle_bindex = angle_bindices[ua_idx]
#             fixed_angle_counts[angle_bindex].append(unbinned_sides[ua_idx])

#     for fa_idx in range(len(fixed_angle_counts)):
#         fixed_theta = np.transpose(fixed_angle_counts[fa_idx], (1, 0))

#         fixed_theta_hist, a_side_edges, b_side_edges = np.histogram2d(*fixed_theta, bins=radii_bins)

#         g3_hist[fa_idx] = np.divide(
#             fixed_theta_hist.T,
#             2
#             * np.pi  # angular average
#             * rho
#             * n_particles
#             * (delta_r * delta_theta) ** 2
#             * np.outer(a_side_edges[1:], b_side_edges[1:]),
#         )

#     return g3_hist


# ==================================================
# NUMPY-OPTIMIZED SAMPLING
# ==================================================


# @nb.njit(fastmath=True, parallel=True)
# def unbinned_angle_sorted_triangles(
#     coords: np.array,
#     lv: np.array,
#     angle_bins: np.array,
#     angle_bin_counts: np.array,
#     crude_triangles: list,
# ) -> None:
#     """
#     DOCSTRING
#     Args
#         coords = (N, d) array of d-dimensional coordinates of all N points
#         lv = (d, d) ndarray of primitive lattice vectors
#         angle_bins = (M, 1) ndarray of discretized bins for angles
#         angle_bin_counts = (M, 1) ndarray of counts in each discretized bin of angles
#         crude_triangles =
#     Returns
#         ...
#     """

#     n_particles, _ = coords.shape

#     for i in range(n_particles - 2):
#         ci_2 = n_particles - i - 1

#         p1 = coords[i]

#         for j in nb.prange(i + 1, n_particles):
#             ci_1 = j - i - 1

#             p2 = coords[j]

#             v12 = jitted_nearest_image(lv, p1, p2)
#             v12_norm = np.linalg.norm(v12)

#             for k in range(j + 1, n_particles):
#                 p3 = coords[k]

#                 v13 = jitted_nearest_image(lv, p1, p3)
#                 v13_norm = np.linalg.norm(v13)

#                 # angle between vectors connecting (i, j) and (i, k)
#                 # checks edge case where |varphi| slightly greater than \pi
#                 varphi = np.nanmin([3.1415, np.arccos(np.dot(v12, v13) / v12_norm / v13_norm)])

#                 ab_idx = np.digitize(varphi, angle_bins)
#                 angle_bin_counts[ab_idx] += 1

#                 # contiguous index
#                 jk = int(ci_1 * (ci_2 - 1) - ci_1 * (ci_1 + 1) / 2 + (k - i - 1) - 1)

#                 crude_triangles[i][jk] = [varphi, v12_norm, v13_norm]

#     return


# @nb.njit(parallel=True)
# def binned_angle_sorted_triangles(
#     clean_triangles: np.array, crude_triangles: np.array, angle_bins: np.array
# ) -> None:
#     """
#     DOCSTRING
#     """

#     for i in nb.prange(len(crude_triangles)):
#         unbinned_triangles = crude_triangles[i]

#         unbinned_angles = unbinned_triangles[:, 0]
#         unbinned_sides = unbinned_triangles[:, 1:]

#         angle_bindices = np.digitize(unbinned_angles, angle_bins)

#         for ua_idx in nb.prange(unbinned_angles.size):
#             angle_bindex = angle_bindices[ua_idx]
#             clean_triangles[angle_bindex][ua_idx] = unbinned_sides[ua_idx]

#     return


# def angle_sorted_triangles(
#     coords: np.array,
#     lv: np.array,
#     angle_bins: np.array,
#     angle_bin_counts: np.array,
#     crude_triangles: list,
# ) -> None:
#     """
#     DOCSTRING
#     Args
#         point_pattern = (N, d) array of d-dimensional coordinates of all N points
#         lv = (d, d) ndarray of primitive lattice vectors
#         angle_bins = (M, 1) ndarray of bins discretized bins for angles
#         crude_triangles = ...
#     Returns
#         clean_triangles = ...
#     """
#     # ts = time.time()
#     # te = time.time()
#     # print(f"elapsed time = {te-ts:0.2e} s")

#     unbinned_angle_sorted_triangles(coords, lv, angle_bins, angle_bin_counts, crude_triangles)

#     clean_triangles = [np.zeros((abc, 2), dtype=np.float64) for abc in angle_bin_counts]

#     binned_angle_sorted_triangles(clean_triangles, crude_triangles, angle_bins)

#     return clean_triangles


# def sample_2d_g3(
#     point_pattern: np.array, lv: np.array, delta_r: float = 2e-2, delta_theta: float = np.pi / 20
# ) -> np.array:
#     """
#     STILL NEEDS TO BE CHECKED!!!!!!!!!!!!!!!!!!

#     Brute force sampling of three-point correlation function g_3(r_12, r_13, theta)
#     Assumes box is anchored at (0, 0) with side lengths (Lx, Ly)
#     Args
#         point_pattern = (N, d) array of d-dimensional coordinates of all N points
#         lv = (d, d) ndarray of primitive lattice vectors
#         delta_r = spacing between successive radii for binning result
#         delta_theta = spacing between successive angles for binning result
#     Returns
#         g3_hist = (T, M, N) ndarray of the three point correlation function
#             axis = 0 (T): indexes the angle of the triangle
#             axis = 1 (M): indexes first side length of the triangle
#             axis = 2 (N): indexes second side length of the triangle
#             (T, M, N) depend on choice of (delta_theta, delta_r, delta_r) and size of simulation box
#     """
#     n_particles, _ = point_pattern.shape
#     vol = parallelipiped_volume(lv)
#     rho = n_particles / vol

#     angle_bins = np.arange(delta_theta, np.pi + delta_theta, delta_theta)
#     angle_bin_counts = np.zeros_like(angle_bins, dtype=np.int32)
#     radii_bins = np.arange(delta_r, np.max(lv), delta_r)

#     g3_hist = np.zeros(
#         (angle_bins.size, radii_bins.size - 1, radii_bins.size - 1), dtype=np.float64
#     )
#     print(f"g3 size = {g3_hist.nbytes / 10**9:0.2e} Gb")

#     unbinned_triangles = [
#         # np.zeros((ld * sd // 2, 3), dtype=np.float64) for (ld, sd) in unbinned_dimensions
#         np.zeros(((n_particles - i - 1) * (n_particles - i - 2) // 2, 3), dtype=np.float64)
#         for i in range(n_particles - 2)
#     ]

#     ##### PRIMARY SUBROUTINE #####
#     ts = time.time()

#     binned_triangles = angle_sorted_triangles(
#         point_pattern, lv, angle_bins, angle_bin_counts, unbinned_triangles
#     )

#     te = time.time()
#     print(f"Primary subroutine took {(te-ts):0.3e} s")

#     # build histogram of g3
#     for bt_idx in range(len(binned_triangles)):
#         fixed_theta = np.transpose(binned_triangles[bt_idx], (1, 0))

#         fixed_theta_hist, a_side_edges, b_side_edges = np.histogram2d(*fixed_theta, bins=radii_bins)

#         g3_hist[bt_idx] = np.divide(
#             fixed_theta_hist.T,
#             2
#             * np.pi  # angular average
#             * rho
#             * n_particles
#             * (delta_r * delta_theta) ** 2
#             * np.outer(a_side_edges[1:], b_side_edges[1:]),  # volume elements
#         )

#     return g3_hist


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"}, edgeitems=100)

    noneq_ics = Path("/home/sd2402/scratch-sd2402/noneq_hs_trajectories/initial_conditions")

    # ic_dir = noneq_ics.joinpath("2d-N_10000-rho_10000-sqlattices")
    # data = list(ic_dir.glob("sql*__*"))

    ic_dir = noneq_ics.joinpath("2D-N_10000-rho_10000-rsa/eta_0.4")
    data = list(ic_dir.glob("RSA*__*"))
    diameter = 2 * np.sqrt(0.2 / 10000 / np.pi)

    ensemble = []
    for d in data:
        _, _, lv, point_pattern = read_cco(d)
        g2 = sample_xd_g2(point_pattern, lv, max_l=10*diameter, delta_r=diameter/1e1)
        ensemble.append(g2)


    ensemble = np.array(ensemble)
    g2 = np.mean(ensemble, axis=0)
    g2[:, 0] /= diameter
    write_xd_data(g2, "g2.txt")

    # ===== plotting  =====
    # _, ax = plt.subplots(layout="constrained")

    # ax.set_xlabel("Radius")
    # ax.set_ylabel("g2(r)")
    # ax.set_title("2D Pair Correlation Function")

    # ax.plot(g2[:, 0], g2[:, 1])

    # plt.show()

# %%