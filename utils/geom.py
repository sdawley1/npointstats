#!/bin/bash
"""
Geometry-related utility functions

Sam Dawley
12/2024
"""

import ctypes

import numpy as np
import numpy.linalg as la

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special


# SPHERES_C = ctypes.cdll.LoadLibrary("/home/sd2402/stats/utils/lib_spheres.so")

# sbj = SPHERES_C.besselJ
# sbj.argtypes = [ctypes.c_double, ctypes.c_double]
# sbj.restype = ctypes.c_double


# def besselJ_c(v: float, x: float) -> float:
#     """Compute bessel function using C backend"""
#     return sbj(v, x)


# besselJ_cwrap = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(besselJ_c)


# def besselJ(v: float, x: float) -> float:
#     """Return vth-order Bessel Function of first kind at x with nterms in series expansion"""
#     return besselJ_cwrap(v, x)


############################################################
# MISCELLANEOUS
############################################################


def abs2(x):
    """Squared Modulus of complex number x"""
    return x.real**2 + x.imag**2


def angular_average(
    Z: np.array, xidx: int = 0, yidx: int = -1, kbinsize: float = 2 * np.pi
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

    Returns:
    --------
        W : np.ndarray, shape (n_waves, 2)
    """
    X = Z[:, xidx]
    Y = Z[:, yidx]
    hmean = kbinsize

    bins = np.arange(np.min(X), np.max(X), hmean)
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


def dot_product(v1, v2) -> float:
    """Dot product of two vectors v1 and v2"""
    return jnp.dot(jnp.asarray(v1), jnp.asarray(v2))


# def fsiv(k: float, R: float, d: int) -> float:
#     """
#     Return Fourier transform of scaled intersection volume alpha_2(k;R)
#     cf. Torquato's 'Hyperuniform States of Matter', pg. 15 Eq. 49
#         for how to check that this func is working (I have and it is)
#     Args
#         k = floating point magnitude of wave vector connecting sphere centroids
#         R = floating point radius of spheres (real space)
#         d = dimensionality
#     Returns
#         floating point scaled intersection volume
#     """
#     return (2 * jnp.sqrt(jnp.pi) / k) ** d * jnp.exp(jax.scipy.special.gammaln(1.0 + d / 2.0)) * besselJ(d / 2, k * R) ** 2


def get_perpendicular_2d_vector(v: np.array) -> np.array:
    """Return unit vector orthogonal to v in two-dimensions"""
    return np.array([-v[1], v[0]]) / np.linalg.norm(v)


def nearest_image(primitive: np.array, p1: np.array, p2: np.array = None) -> np.array:
    """
    Compute d-dimensional Euclidean distance between points p1, p2 using nearest-image convention
    Assumes center of primitive lattice is at CORNER of parallelogram defined by primitive
    Args
        p1 = coordinates of tagged point
        p2 = coordinates of other point
        primitive = primitive lattice vectors
    """
    dimension = p1.size
    primitive_norms = map(la.norm, primitive)

    # Nearest-image position of a single particle
    if p2 is None:
        for i in range(dimension):
            k = next(primitive_norms)

            if p1[i] >= k:
                p1[i] -= k

            elif p1[i] <= 0:
                p1[i] += k

        return p1

    # Nearest-image distance between two particles
    dr = p1 - p2
    for i in range(dimension):
        k = next(primitive_norms)

        if dr[i] >= k / 2:
            dr[i] -= k

        elif dr[i] <= -k / 2:
            dr[i] += k

    return np.asarray(dr)


@jax.jit
def _wrap_position(pnorms: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
    """Wrap a single particle position into the primary cell."""
    p1 = jnp.where(p1 >= pnorms, p1 - pnorms, p1)
    p1 = jnp.where(p1 <= 0, p1 + pnorms, p1)
    return p1


@jax.jit
def _nearest_image_dr(pnorms: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """Compute nearest-image displacement between two particles."""
    dr = p1 - p2
    dr = jnp.where(dr >= pnorms / 2, dr - pnorms, dr)
    dr = jnp.where(dr <= -pnorms / 2, dr + pnorms, dr)
    return dr


def jitted_nearest_image(primitive: np.array, p1: np.array, p2: np.array = None) -> jnp.ndarray:
    """
    Compute d-dimensional Euclidean distance between points p1, p2 using nearest-image convention
    Assumes center of primitive lattice is at CORNER of parallelogram defined by primitive
    Args
        p1 = coordinates of tagged point
        p2 = coordinates of other point
        primitive = primitive lattice vectors
    """
    pnorms = jnp.linalg.norm(jnp.asarray(primitive), axis=1)
    if p2 is None:
        return _wrap_position(pnorms, jnp.asarray(p1))
    return _nearest_image_dr(pnorms, jnp.asarray(p1), jnp.asarray(p2))


@jax.jit
def normed_vector_array(a: jnp.ndarray) -> jnp.ndarray:
    """Return array containing norms of all vectors contained in a"""
    return jnp.linalg.norm(a, axis=1)


@jax.jit
def parallelipiped_volume(basis_vectors: jnp.ndarray) -> float:
    """Return volume of d-dimensional parallelipiped with known basis vectors"""
    return jnp.abs(jnp.linalg.det(basis_vectors))


def sphere_volume(R: float, d: int) -> float:
    """Compute volume of d-dimensional sphere of radius R"""
    return jnp.pi ** (d / 2) * R**d / jnp.exp(jax.scipy.special.gammaln(1.0 + d / 2.0))


def sphere_surface(R: float, d: int) -> float:
    """Compute surface area of d-dimensional sphere of radius R"""
    return (2 * jnp.pi ** (d / 2) / jnp.exp(jax.scipy.special.gammaln(d / 2.0))) * R ** (d - 1)


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


def stringify(V: np.array, new_d: int = 4) -> list:
    """
    Stringify V lattice vectors to 4D (a la CCOptimization) for writing to output files
    Args
        V = (n, d) lattice vectors
        new_d = 'dimension' of output basis vectors; must be <=4
    Returns
        res = (1, d) list of strings containing lattice vectors
    """
    res = []
    for v in V:
        line = ""
        svector = list(map(lambda x: f"{x:0.6f}", v))
        if len(svector) < new_d:
            svector += ["0.0" for _ in range(new_d - len(svector))]
        svector = np.asarray(svector).ravel()
        for sv in svector:
            line += sv + "\t"
        res.append(line)
    return res


@jax.jit
def trapz_integrate(X: jnp.ndarray, Y: jnp.ndarray):
    """
    Numerical integration of Y over X using Simpson's rule
    Args
        X, Y = same-size arrays containing independent, dependent data to integrate over
    Returns
        I = floating point numerical approximation to integral
    """
    h = jnp.concatenate([jnp.zeros(1), jnp.diff(X)])
    N = X.shape[0]

    I = h[0] * Y[0] / 3 + h[N - 1] * Y[N - 1] / 3

    even_idx = jnp.arange(2, N - 2, 2)
    odd_idx = jnp.arange(1, N - 1, 2)

    I = I + 2 * jnp.sum(h[even_idx] * Y[even_idx]) / 3
    I = I + 4 * jnp.sum(h[odd_idx] * Y[odd_idx]) / 3

    return I


@jax.jit
def vector_abs_value(a: jnp.ndarray) -> jnp.ndarray:
    """Return absolute value of elements from array a"""
    return jnp.abs(a)


def vector_norm(a, p: int = 2) -> float:
    """Lp vector norm"""
    return jnp.linalg.norm(jnp.asarray(a), ord=p)


if __name__ == "__main__":
    import sys
    import time

    # pylint: disable=E0401
    sys.path.append("/home/sd2402/n_point_statistics/")

    from src.gpu_analysis.g_structure_factor import genkvecs

    d = 3
    # primitive lattice vectors
    v1 = np.array([9.86123741e-01, -1.82682008e-03, -1.04632614e-04])
    v2 = np.array([-1.8269217e-03, 9.8693800e-01, -7.9284022e-05])
    v3 = np.array([-1.0435715e-04, -7.7646073e-05, 9.8493093e-01])
    V = np.array([v1, v2, v3])
    # Arbitrary vectors for testing
    a = np.array([0.44134, 0.23421, 0.81735])
    b = np.array([0.26783, 0.99227, 0.51394])
    c = np.array([1.55321, 0.85592, 0.12345])
    e = np.array([1.15321, 0.85592, 0.12345])
    A = np.array([a, e, b])

    v4 = np.array([1.0, 0.0, 0.0])
    v5 = np.array([0.0, 1.0, 0.0])
    v6 = np.array([0.0, 0.0, 1.0])
