#!/bin/bash
"""
Compute n-point statistics

Sam Dawley
12/2024
"""

import math

import numpy as np
from numba import cuda


def dist_matrix_xd(
    arr1: np.array, arr2: np.array, lv: np.array, tpb: tuple = (32, 32), self_dist: float = np.inf
) -> np.array:
    """
    Driver for computing pairwise distances between two coordinate arrays
    Returns (M, N) distance matrix between coordinates specified by arr1 and arr2
    Uses nearest image convention to implement periodic boundary conditions
    Args
        arr1 (point_pattern) = (M, d) array of d-dimensional coordinates of M points
        arr2 (point_pattern) = (N, d) array of d-dimensional coordinates of N points
        lv = (d, d) ndarray of primitive lattice vectors
        tpb = tuple of threads_per_block to use for GPU kernel
    Returns
        radii = (N, M) ndarray of pairwise distances between coordinates in arr1, arr2
    """
    n_particles_1, dimension_1 = arr1.shape
    n_particles_2, dimension_2 = arr2.shape
    radii = np.zeros((n_particles_1, n_particles_2))  # arr1 (arr2) indexed by rows (columns)

    if dimension_1 != dimension_2:
        print(f"Arrays arr1 and arr2 have inconcomitant shapes: {arr1.shape} and {arr2.shape}...")
        return radii

    # specify blocks_per_grid to use for GPU kernel
    # always using 2D grid on GPU since we're interested in all pairwise distances of N particles
    bpg_x = int(np.ceil(radii.shape[0] / tpb[0]))
    bpg_y = int(np.ceil(radii.shape[1] / tpb[1]))
    bpg = (bpg_x, bpg_y)

    g_dists = cuda.to_device(radii)
    g_coords_1 = cuda.to_device(arr1)
    g_coords_2 = cuda.to_device(arr2)
    g_lv = cuda.to_device(lv)

    if dimension_1 == 2:
        dist_matrix_2d[bpg, tpb](g_dists, g_coords_1, g_coords_2, g_lv)
    elif dimension_1 == 3:
        dist_matrix_3d[bpg, tpb](g_dists, g_coords_1, g_coords_2, g_lv)

    radii = g_dists.copy_to_host()
    np.fill_diagonal(radii, self_dist)

    return radii


@cuda.jit(fastmath=True)
def dist_matrix_2d(
    pairwise_distances: np.array, arr1: np.array, arr2: np.array, primitive: np.array
) -> None:
    """
    Compute pairwise distances between two coordinate arrays
        and populate 'pairwise_distances'
    Uses nearest image convention to implement periodic boundary conditions
    Args
        R = (M, N) array of all pairwise distances (initially empty)
        arr1 (point_pattern) = (M, d) array of d-dimensional coordinates of M points
        arr2 (point_pattern) = (N, d) array of d-dimensional coordinates of N points
        primitive = (d, d) ndarray of primitive lattice vectors
    """
    i, j = cuda.grid(ndim=2)
    if i >= pairwise_distances.shape[0] or j >= pairwise_distances.shape[1]:
        return

    # compute nearest image distances
    cx = math.hypot(primitive[0][0], primitive[0][1])
    cy = math.hypot(primitive[1][0], primitive[1][1])
    dx = arr1[i][0] - arr2[j][0]
    dy = arr1[i][1] - arr2[j][1]

    if dx >= cx / 2:
        dx -= cx
    elif dx <= -cx / 2:
        dx += cx

    if dy >= cy / 2:
        dy -= cy
    elif dy <= -cy / 2:
        dy += cy

    # pylint: disable=E1121
    cuda.atomic.add(pairwise_distances, (i, j), math.hypot(dx, dy))

    return


@cuda.jit(fastmath=True)
def dist_matrix_3d(
    pairwise_distances: np.array, arr1: np.array, arr2: np.array, primitive: np.array
) -> None:
    """
    Compute pairwise distances between two coordinate arrays
        and populate 'pairwise_distances'
    Uses nearest image convention to implement periodic boundary conditions
    Args
        R = (M, N) array of all pairwise distances (initially empty)
        arr1 (point_pattern) = (M, d) array of d-dimensional coordinates of M points
        arr2 (point_pattern) = (N, d) array of d-dimensional coordinates of N points
        primitive = (d, d) ndarray of primitive lattice vectors
    """
    i, j = cuda.grid(ndim=2)
    if i >= pairwise_distances.shape[0] or j >= pairwise_distances.shape[1]:
        return

    cx = math.hypot(primitive[0][0], primitive[0][1])
    cy = math.hypot(primitive[1][0], primitive[1][1])
    cz = math.hypot(primitive[2][0], primitive[2][1])

    dx = arr1[i][0] - arr2[j][0]
    dy = arr1[i][1] - arr2[j][1]
    dz = arr1[i][2] - arr2[j][2]

    if dx >= cx / 2:
        dx -= cx
    elif dx <= -cx / 2:
        dx += cx

    if dy >= cy / 2:
        dy -= cy
    elif dy <= -cy / 2:
        dy += cy

    if dz >= cz / 2:
        dz -= cz
    elif dz <= -cz / 2:
        dz += cz

    hypot = math.sqrt(dx**2 + dy**2 + dz**2)

    # pylint: disable=E1121
    cuda.atomic.add(pairwise_distances, (i, j), hypot)

    return


if __name__ == "__main__":
    ...
