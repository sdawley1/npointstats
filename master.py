#!/bin/bash
"""
Primary interface for computing n-point correlation functions

Anaconda setup:
    $ conda create --name gpustats
    $ pip install numpy numba h5py matplotlib

Sam Dawley
12/2024
"""

import argparse
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Tuple

# pylint: disable=E0401
sys.path.append("/home/sd2402/n_point_statistics/")

import numpy as np

from utils.point_pattern import Ensemble
from utils.readwrite import read_cco, read_dnv, read_lstj, read_morsefire

# ==================================================
# UTILITIES
# ==================================================

def read_file(inpfile: str):
    """Open and read-in lines from an input file"""
    with open(inpfile, "r", encoding="utf-8") as f:
        yield from f


def read_simple_data(ifile: str, delim: str = "\t") -> Tuple[None, None, np.ndarray, np.ndarray]:
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
    lv = np.array([[1, 0], [0, 1]], dtype=np.float64)
    for line in read_file(ifile):
        if "#" in line:
            header += line
            continue
        l = line.strip("\n").split(delim)
        res = [float(a) for a in l[:]]
        z.append(res)
    return None, None, lv, np.asarray(z)


def read_chiral_active(ifile: str, delim: str = " ") -> np.array:
    """
    Read data from `ifile` of the form
        x_1,y_1,...,z_1
        x_2,y_2,...,z_2
        x_3,y_3,...,z_3, etc.
    Returns
        z = (N, X) ndarray of data
    """
    z = []
    header = ""
    dimension = 2

    # lv = np.array([[114.4114, 0.0], [0.0, 114.4114]], dtype=np.float64)
    lv = np.array([[161.80216, 0.0], [0.0, 161.80216]], dtype=np.float64)

    for line in read_file(ifile):
        if "#" in line: 
            header += line
            continue
        l = line.strip("\n").split(delim)
        res = [float(a) for a in l[:dimension]]
        z.append(res)

    return None, None, lv, np.asarray(z)


# ==================================================
# ENSEMBLES
# ==================================================


def read_ensemble(dp: Path, file_names: str = "*__*", parser=read_cco) -> list:
    """
    Read in ensemble of point patterns
    """
    patterns = []

    if parser.__name__ == "read_morsefire":
        _, _, primitive, coordinates = read_morsefire(dp)
        patterns.append((primitive, coordinates))

        return patterns

    for ifile in dp.glob(file_names):
        print(f"Reading file {ifile}", flush=True)
        _, _, primitive, coordinates = parser(ifile)
        patterns.append((primitive, coordinates))

    return patterns


def ensemble_statistics(TheEnsemble: Ensemble, stats: list) -> int:
    """
    Compute n-point statistics for ensemble of point patterns
    """
    for st in stats:
        TheEnsemble.compute_statistic(st)
        # TheEnsemble.parallel_compute_statistic(st)
        TheEnsemble.write_statistic(st)
        # TheEnsemble.plot_statistic(st)

    return 0


# ==================================================
# PRIMARY SUBROUTINE
# ==================================================


def parse_input(args, parser=read_cco) -> int:
    """
    Primary subroutine for computing desired n-point statistics
    Args
        args = list of arguments retured from argparse
    """
    statistics = args.statistics
    file_names = args.name
    if file_names[-1] != "*":
        file_names += "*"

    data_path = Path(args.directory)

    patterns = read_ensemble(data_path, file_names=file_names, parser=parser)
    Omega = Ensemble(patterns)
    ensemble_statistics(Omega, statistics)

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="n-pt Statistics",
        usage="$ python3 runjob.py -d <input_directory:optional> -s <statistics:optional> -f <hdf5file:optional> -n <name:optional>",
    )

    parser.add_argument(
        "-d",
        "--directory",
        action="store",
        nargs="?",
        default="./patterns",
        help="Input directory containing ensemble of point patterns",
        required=False,
    )

    parser.add_argument(
        "-f",
        "--hdf5",
        action="store",
        default="./intermediate_outputs.hdf5",
        help="Input HDF5 file containing trajectory of points patterns",
        required=False,
    )

    parser.add_argument(
        "-s", "--statistics", nargs="+", help="Statistic(s) to compute", required=False
    )

    parser.add_argument(
        "-n",
        "--name",
        action="store",
        default="__",
        help="Name(s) of files to read",
        required=False,
    )

    parser.add_argument(
        "-p",
        "--fileparser",
        action="store",
        default="cco",
        choices=["cco", "dnv", "lstj", "morsefire", "simple", "cas"],
        help="Name(s) of files to read",
        required=False,
    )

    # handle the file parser
    parsing = {
        "cco": read_cco,
        "dnv": read_dnv,
        "lstj": read_lstj,
        "morsefire": read_morsefire,
        "simple": read_simple_data,
        "cas": read_chiral_active
    }

    args = parser.parse_args()
    file_parser = parsing.get(args.fileparser, None)

    if isinstance(file_parser, type(None)):
        print(f"ERROR: Could not associate -p '{args.fileparser}' with a file parser.", flush=True)
        sys.exit(1)

    parse_input(args, parser=file_parser)
