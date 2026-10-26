#!/bin/bash
"""
File I/O

Sam Dawley
12/2024
"""

import glob
from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np


# ==================================================
# AUXILIARY FUNCTIONS
# ==================================================


def stringify(v: np.array, new_d: int = 4) -> list:
    """
    Stringify V lattice vectors to 4D (a la CCOptimization) for writing to output files
    Args
        V = (n, d) lattice vectors
        new_d = 'dimension' of output basis vectors; must be <=4
    Returns
        res = (1, d) list of strings containing lattice vectors
    """
    res = []

    for vv in v:
        line = ""
        svector = list(map(lambda x: f"{x:0.6f}", vv))

        if len(svector) < new_d:
            svector += ["0.0" for _ in range(new_d - len(svector))]

        svector = np.asarray(svector).ravel()

        for sv in svector:
            line += sv + "\t"
        res.append(line)

    return res


# ==================================================
# READING
# ==================================================


def read_file(inpfile: str):
    """Open and read-in lines from an input file"""
    with open(inpfile, "r", encoding="utf-8") as f:
        yield from f


def read_dir(inpdir: str, containing: str = ""):
    """
    Open and read-in lines from all input files within a directory
    'containing' parameter optional to specify substring of
    filenames which should be read from
    """
    if inpdir[-1] != "/":
        inpdir += "/"
    return glob.glob(inpdir + containing)


def read_cco(ifile: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read data from CCO, taking advantage of fixed file setup

    Returns:
    --------
        d : int
            system dimensionality
        num_particles : int
            number of particles
        lattice_vectors : np.ndarray
            primitive lattice vectors of 3D system
        coordinates : np.ndarray
            positions of all individual spheres
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


def read_dnv(ifile: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read data from Donev, taking advantage of fixed file setup

    Parameters:
    -----------
        ifile : str
            name of/path to input file

    Returns:
    --------
        d : int
            dimensionality of system
        lattice_vectors : np.ndarray, shape (d, d)
            primitive lattice vectors
        centroids : np.ndarray, shape (N, d)
            centroids of all N hyperspheres in packing
    """
    procline = lambda l: l.strip("\n").split()

    centroids = []

    with open(ifile, "r", encoding="utf-8") as f:
        lines = f.readlines()
        try:
            d = int(lines[0].strip("\n"))
        except ValueError as e:
            print(f"Error: {e}", flush=True)
            d = int(lines[0].strip("\n").split()[0])

        if d > 3:
            print(f"Cannot process systems of dimension = {d} > 3. Exiting...")
            sys.exit(1)

        n_particles, n_species = map(int, procline(lines[1]))
        n_particles_per_species = [int(Ns) for Ns in procline(lines[2])]
        diameter_per_species = [float(D) for D in procline(lines[3])]
        # lattice_vectors = np.asarray([int(l) for l in procline(lines[4])]).reshape(d, d)
        lattice_vectors = np.array([float(l) for l in procline(lines[4])]).reshape(d, d)
        pbcs = "T" in lines[5].split(" ")

        for j in range(6, len(lines)):
            coord = lines[j].strip("\n").split()
            centroids.append([float(a) for a in coord])

    return d, n_particles, lattice_vectors, np.asarray(centroids)


def read_lstj(ifile: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Read data from LSTJ, taking advantage of fixed file setup

    Parameters:
    -----------
        ifile : str
            name of/path to input file

    Returns:
    --------
        d : int
            dimensionality of system
        n_particles : int
            number of sphere centroids
        lattice_vectors : np.ndarray, shape (d, d)
            primitive lattice vectors
        centroids : np.ndarray, shape (N, d)
            centroids of all N hyperspheres in packing
    """
    procline = lambda l: l.strip("\n").split()

    centroids = []

    with open(ifile, "r", encoding="utf-8") as f:
        lines = f.readlines()
        d = int(lines[0].strip("\n").split()[0])

        if d > 3:
            print(f"Cannot process systems of dimension = {d} > 3. Exiting...")
            sys.exit(1)

        n_particles, n_species = map(int, procline(lines[1]))
        n_particles_per_species = [int(Ns) for Ns in procline(lines[2])]
        diameter_per_species = [float(D) for D in procline(lines[3])]

        lattice_vectors = []
        for lidx in range(4, 4 + d, 1):
            lattice_vectors.append([float(l) for l in procline(lines[lidx])])
        lattice_vectors = np.asarray(lattice_vectors).reshape(d, d)
        pbcs = "T" in lines[5 + d].split(" ")

        for j in range(6 + d, len(lines)):
            coord = procline(lines[j])
            centroids.append([float(a) for a in coord])

    rho = n_particles / np.linalg.det(lattice_vectors)
    print(f"density = {rho:0.2e}")
    return d, n_particles, lattice_vectors, np.asarray(centroids)


def read_xd_data(ifile: str, delim: str = ",") -> np.array:
    """
    Read data from `ifile` of the form
        x_1,y_1,...,z_1
        x_2,y_2,...,z_2
        x_3,y_3,...,z_3, etc.
    Returns
        z = (N, X) ndarray of data
    """
    header = ""
    z = []
    for line in read_file(ifile):
        if "#" in line:
            header += line
            continue
        l = line.strip("\n").split(delim)
        res = [float(a) for a in l[:]]
        z.append(res)
    return np.asarray(z)


# ==================================================
# FIRE OPTIMIZATION OUTPUTS A LA PETER
# ==================================================


def read_morse_ndarray(inpfile: Path) -> np.ndarray:
    config = []
    with open(inpfile, "r", encoding="utf-8") as f:
        for line in f.readlines():
            config.append(np.fromstring(line, sep=" "))
    return np.asarray(config, dtype=np.float64)


def read_morse_scalars(inpfile: Path) -> Dict[str, float]:
    scalars = {}
    with open(inpfile, "r", encoding="utf-8") as f:
        for line in f.readlines():
            key, value = line.strip("\n").split("\t")
            scalars[key] = float(value)
    return scalars


def read_morsefire(root_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:

    # ----- collect data -----

    for d in root_dir.glob("*"):
        if "boxSize.dat" in str(d):
            boxsize = read_morse_ndarray(d)
        elif "positions.dat" in str(d):
            positions = read_morse_ndarray(d)
        elif "radii.dat" in str(d):
            radii = read_morse_ndarray(d)
        elif "scalars.dat" in str(d):
            scalars = read_morse_scalars(d)

    # ----- reformat -----

    dimension = int(scalars["nDim"])
    n_particles = int(scalars["numParticles"])

    lattice_vectors = np.zeros((dimension, dimension), dtype=np.float64)
    lattice_vectors[0, 0] = boxsize[0]
    lattice_vectors[1, 1] = boxsize[1]

    return dimension, n_particles, lattice_vectors, positions


def morsefire_to_cco(
    root_dir: Path, boxsize: np.ndarray, positions: np.ndarray, scalars: Dict[str, float]
) -> str:
    """
    Rewrite the gathered data to a format suitable for the ConfigPack manager.
    """
    d = int(scalars["nDim"])
    N = int(scalars["numParticles"])

    lv = np.zeros((d, d), dtype=np.float64)
    lv[0, 0] = boxsize[0]
    lv[1, 1] = boxsize[1]

    outfile = "cco-positions.dat"
    write_cco(lv, positions, root_dir.joinpath(outfile))

    return outfile


# ==================================================
# WRITING
# ==================================================


def write_cco(lv: np.array, coords: np.array, outfile: str) -> None:
    """
    Write files in format used for CCO input
    Args
        lv = (d, d) primitive lattice vectors of cell
        coords = (N, d) d-dimensional coordinates of all points in cell
    """
    # metadata
    d = lv.shape[0]

    print(f"Writing to ./{outfile}")
    print(f"Coordinate array has shape {coords.shape} and lattice vectors has shape {lv.shape}")
    with open(outfile, "w") as f:
        # write dimension
        f.write(f"{d}\n")

        # write primitive lattice vectors
        sv = stringify(lv)
        for v in sv:
            f.write(f"{v}\n")

        # write coordinates
        for pt in coords:
            spt = stringify([pt])[0]
            f.write(f"{spt}\n")
    return 0


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
            # ----- header -----
            if header:
                hlines = header.split("\n")
                for hl in hlines:
                    f.write("# " + hl + "\n")

            # ----- data -----
            for zz in z:
                s = delim.join(map(str, zz)) + "\n"
                f.write(s)

        print(f"Written to {ofile}...")
        return True

    except:
        print("Output unsuccessfully written")
        return False


if __name__ == "__main__":
    ...
