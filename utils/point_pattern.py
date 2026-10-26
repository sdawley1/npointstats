#!/bin/bash
"""
Class for storing and computing 2-point statistics
The only information required by the user is
    1. a d-dimensional point pattern of N particles in Euclidean space, and
    2. the primitive lattice vectors which define the region in which points are contained

12/2024
Sam Dawley
"""

from collections import namedtuple
from functools import partial
import glob
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

import matplotlib.pyplot as plt

# pylint: disable=E0401cond
sys.path.append("/home/sd2402/n_point_statistics/")

# from src.cpu_analysis.c_number_variance import sample_xd_nv
# from src.cpu_analysis.c_npt_correlation import sample_2d_g3

from src.cpu_analysis.c_structure_factor import sample_xd_sf_collective as sample_xd_sf
# from src.cpu_analysis.c_structure_factor import sample_xd_sf
from src.cpu_analysis.c_npt_correlation import sample_xd_g2

from utils.geom import angular_average
from utils.readwrite import read_cco, write_xd_data

EnsembleMember = namedtuple("Member", ["lv", "coords"])
StatContainers = namedtuple("Statistic", ["desc", "ufunc"])


# ==================================================
# UTILITIES
# ==================================================


def compute_member_statistic(ensemble_member: EnsembleMember, ufunc: callable) -> np.ndarray:
    """
    Wrapper function to compute statistic for a single member.
    Must be at module level for pickling.
    """
    lv, coords = ensemble_member
    
    ts = time.time()
    z = ufunc(coords, lv)  # returns ndarray

    print(f"elapsed time = {(time.time()-ts):0.4f}", flush=True)
    
    return z

def plot_xd_data(st: str, x: np.ndarray, y: np.ndarray, outfile: Path) -> None:
    """
    Parameters:
    -----------
        st : str
            statistic we wish to plot
        x : np.ndarray
            x-coordinates of the data
        y : np.ndarray
            y-coordinates of the data
    """
    fig, ax = plt.subplots()

    if st == "sf":
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$S(k)$")
    elif st == "g2":
        ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$g_2(r)$")
    else:
        print(f"Statistic '{st}' not supported")
        sys.exit(1)

    ax.plot(x, y, linestyle="-", color="black")

    plt.savefig(outfile, dpi=512)
    print(f"Saved figure to {outfile}", flush=True)


# ==================================================
# ENSEMBLE CLASS
# ==================================================


class Ensemble:
    """
    Class for storing ensemble of point patterns
    and computing basic (pair) statistics
    """

    # ===== INITIALIZATION ====

    def __init__(self, members: list = []) -> None:
        """
        Initialize container to store ensemble of point patterns
        Args
            members = list of tuples of form (lv, coords)
                lv = primitive lattice vectors of box containing configuration
                coords = (N, d) d-dimensional point pattern containing N points
        """
        self.members = members
        self.n_members = len(members)

        # functions responsible for computing various statistical descriptors
        self.stat_funcs = {
            # "nv": StatContainers("Local number variance", sample_xd_nv),
            "g2": StatContainers("Pair correlation function", sample_xd_g2),
            # "g3": StatContainers("Triplet correlation function", sample_2d_g3),
            # "nS": StatContainers("Local number variance", compute_2d_nv),
            "sf": StatContainers("Structure Factor", sample_xd_sf),
        }

        # empty list for each stat for separate ensemble and angular averaged results
        self.stat_results = {k: [None, None] for k in self.stat_funcs}

    def __repr__(self) -> str:
        res = ""
        for midx in range(self.n_members):
            _, coords = self.members[midx]
            n_particles, d = coords.shape
            res += f"Member {midx:>3d} w/ {n_particles:>5d} points in {d:>2d}-dimensions\n"
        return res

    # ===== PRIVATE FUNCTIONS ====

    def _add_member(self, lv: np.array, coords: np.array) -> int:
        """Add member to ensemble"""
        self.members.append(EnsembleMember(lv, coords))
        self.n_members += 1
        return 0

    def load_cco(self, f: str) -> int:
        """Load member from CCO text file"""
        _, _, lv, coords = read_cco(f)
        self._add_member(lv, coords)
        return 0

    # ===== COMPUTING AND WRITING STATISTICS ====

    def compute_statistic(self, stat: str) -> None:
        """
        Compute desired statistics

        Parameters:
        -----------
            stat : str
                string of statistic we wish to compute; takes values of
                'g2' = pair correlation function
                'sf' = structure factor
                'nv' = number variance via Monte Carlo Sampling
                'nS' = number variance via an integral over the structure factor
                'xv' = spectral density
        """
        # dimension = None

        if stat not in self.stat_funcs:
            print(f"Requested statistic '{stat}' not supported")
            sys.exit(1)

        ufunc = self.stat_funcs[stat].ufunc
        print(f"{ufunc.__name__=}")

        eres = []
        for midx in range(self.n_members):
            lv, coords = self.members[midx]

            print(f"Member {midx:>3d} (N={coords.shape[0]:d})...", end=" ")
            ts = time.time()
            z = ufunc(coords, lv)  # returns ndarray
            te = time.time()
            print(f"elapsed time = {(te-ts):0.4f}", flush=True)

            eres.append(z)

        eres = np.asarray(eres)
        ensemble_avg = np.mean(eres, axis=0)

        self.stat_results[stat][0] = ensemble_avg

        # extra step for statistics aren't already angular averaged
        if stat == "sf":
            # delta_k = 250 * np.pi
            delta_k = np.pi / 100

            # if dimension == 1:
            #     delta_k *= 40

            angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1, kbinsize=delta_k)
            self.stat_results[stat][1] = angular_avg

        return 0


    def parallel_compute_statistic(self, stat: str, n_processes=None) -> None:
        """
        Compute desired statistics (parallelized version)
        Parameters:
        -----------
            stat : str
                string of statistic we wish to compute
            n_processes : int, optional
                number of parallel processes (default: number of CPU cores)
        """
        self._check_cuda()
        
        if stat not in self.stat_funcs:
            print(f"Requested statistic '{stat}' not supported")
            sys.exit(1)
        
        ufunc = self.stat_funcs[stat].ufunc
        print(f"{ufunc.__name__=}")
        
        # Create partial function with fixed arguments
        partial_compute_statistic = partial(
            compute_member_statistic,
            ufunc=ufunc,
        )
        
        # Parallel processing
        with mp.Pool(processes=n_processes) as pool:
            eres = pool.map(partial_compute_statistic, self.members)
        
        eres = np.asarray(eres)
        ensemble_avg = np.mean(eres, axis=0)
        self.stat_results[stat][0] = ensemble_avg
        
        # Extra step for statistics that aren't already angular averaged
        if stat == "sf":
            # delta_k = np.pi / 120 # for smallest Da
            delta_k = np.pi / 20
            angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1, kbinsize=delta_k)
            self.stat_results[stat][1] = angular_avg
        
        return 0

    def write_statistic(self, stat: str) -> None:
        """
        Write desired statistic to output file
        Args
            stat = string taking values of 'nv', 'g2', 'nS', 'sf',
        """
        if self.stat_results[stat][0] is not None:
            write_xd_data(
                self.stat_results[stat][0], f"{stat}_res.txt", header=self.stat_funcs[stat].desc
            )

            write_xd_data(
                self.stat_results[stat][1], f"{stat}_bin.txt", header=self.stat_funcs[stat].desc
            )

        else:
            print(f"Desired statistic '{stat}' not yet computed; performing calculation...")

            self.compute_statistic(stat)

            write_xd_data(
                self.stat_results[stat][0], f"{stat}_res.txt", header=self.stat_funcs[stat].desc
            )

            write_xd_data(
                self.stat_results[stat][1], f"{stat}_bin.txt", header=self.stat_funcs[stat].desc
            )

        return 0

    def plot_statistic(self, stat: str) -> None:
        """
        Plot desired statistic
        Args
            stat = string taking values of 'nv', 'g2', 'nS', 'sf',
        """
        if self.stat_results[stat][0] is not None:
            # plot_xd_data(stat, self.stat_results[stat][0], Path(f"{stat}_res.png"))
            plot_xd_data(stat, *self.stat_results[stat][1].T, Path(f"./{stat}_bin.png"))


if __name__ == "__main__":

    # metadata
    Omega = Ensemble()

    CONFIG_DIR = "/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/patterns/*"
    files = glob.glob(CONFIG_DIR)

    for fl in files:
        Omega.load_cco(fl)

    # Omega.compute("g2")
    # Omega.write_statistic("g2")
    # Omega.write_statistic("nv")
    # Omega.write_statistic("nS")
