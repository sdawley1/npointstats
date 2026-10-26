#!/bin/bash
"""
Class for storing and computing 2-point statistics for trajectories to stealthy ground states
The only information required by the user is
    1. a d-dimensional point pattern of N particles in Euclidean space, and
    2. the primitive lattice vectors which define the region in which points are contained

1/2024
Sam Dawley
"""

import sys
from collections import namedtuple
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=E0401cond
sys.path.append("/home/sd2402/n_point_statistics/")

from src.cpu_analysis.c_number_variance import sample_xd_nv
from src.cpu_analysis.c_npt_correlation import sample_2d_g3

from src.gpu_analysis.g_npt_correlation import sample_xd_g2
from src.gpu_analysis.g_number_variance import compute_2d_nv
from src.gpu_analysis.g_structure_factor import sample_xd_sf

from utils.geom import angular_average
from utils.readwrite import read_cco, write_xd_data

EnsembleMember = namedtuple("Member", ["lv", "coords"])
TrajectoryMember = namedtuple("Trajectory", ["lv", "trajectory"])
StatContainers = namedtuple("Statistic", ["desc", "ufunc"])

STAT_FUNCS = {
    "nv": StatContainers("Local number variance", sample_xd_nv),
    "g2": StatContainers("Pair correlation function", sample_xd_g2),
    "g3": StatContainers("Triplet correlation function", sample_2d_g3),
    "nS": StatContainers("Local number variance", compute_2d_nv),
    "sf": StatContainers("Structure Factor", sample_xd_sf),
    # "xv": StatContainers("Spectral Density", sample_xd_xv),
}


##################################################
# Abstract Base class for Ensemble
##################################################


class Ensemble(ABC):
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
        self.stat_results = {k: [None, None] for k in STAT_FUNCS}
        return

    def __repr__(self) -> str:
        return ""

    @abstractmethod
    def _add_member(self, lv: np.array, point_pattern: np.array) -> None:
        """Add member to ensemble"""
        self.members.append(EnsembleMember(lv, point_pattern))
        self.n_members += 1
        return 0

    @abstractmethod
    def compute_statistic(self, stat: str) -> None:
        """Compute ensemble-average of desired statistic"""
        return

    @abstractmethod
    def write_statistic(self, stat: str) -> None:
        """Write ensemble-average of desired statistic to file"""
        return


##################################################
# 'Standard' Ensemble of Point Patterns
##################################################


class PointPatterns(Ensemble):
    """
    Class for storing ensemble of point patterns
    and computing basic (pair) statistics
    """

    def __init__(self, members: list = []) -> None:
        return super().__init__(members)

    def __repr__(self) -> None:
        res = ""
        for midx in range(self.n_members):
            _, coords = self.members[midx]
            n_particles, d = coords.shape
            res += f"Member {midx:>3d}: {d:>2d}D configuration of {n_particles:>5d} points\n"
        return res

    def _add_member(self, lv: np.array, coords: np.array) -> None:
        return super()._add_member(lv, coords)

    def load_cco(self, f: str) -> None:
        """Load member from CCO text file"""
        _, _, lv, coords = read_cco(f)
        self._add_member(lv, coords)
        return 0

    def compute_statistic(self, stat: str) -> None:
        """
        Args
            stat = string of statistic we wish to compute; takes values of
                'g2' = pair correlation function
                'sf' = structure factor
                'nv' = number variance via Monte Carlo Sampling
                'nS' = number variance via an integral over the structure factor
                'xv' = spectral density
        """
        if stat not in self.stat_funcs:
            print(f"Requested statistic '{stat}' not supported")
            sys.exit(1)

        ufunc = self.stat_funcs[stat].ufunc

        eres = []
        for midx in range(self.n_members):
            print(f"Member {midx:d}...")
            ensemble_member = self.members[midx]

            lv = ensemble_member.lv
            coords = ensemble_member.coords

            z = ufunc(coords, lv)  # returns ndarray
            eres.append(z)

        eres = np.asarray(eres)
        ensemble_avg = np.mean(eres, axis=0)

        self.stat_results[stat][0] = ensemble_avg

        # extra step for statistics aren't already angular averaged
        if stat == "sf":
            angular_avg = angular_average(ensemble_avg, xidx=-2, yidx=-1)
            self.stat_results[stat][1] = angular_avg

        return 0

    # def plot_statistics(self, stat: str, **kwargs) -> None:
    #     """
    #     Simple plotting procedure to look at a desired statistic
    #     """
    #     if stat == "g2":
    #         xlabel, ylabel = "r", "g2(r)"
    #         xlim, ylim = [0, 8], [0, 4]
    #     if stat == "sf":
    #         xlabel, ylabel = "k", "S(k)"
    #         xlim, ylim = [0, 20], [0, 4]

    #     print(stat, self.stat_results[stat])
    #     x_data, y_data = np.transpose(self.stat_results[stat][0])

    #     # plotting
    #     _, ax = plt.subplots(layout="constrained")

    #     ax.plot(x_data, y_data, **kwargs)

    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_xlim(*xlim)
    #     ax.set_ylim(*ylim)

    #     plt.show()

    #     return 0

    def write_statistic(self, stat: str) -> None:
        """
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


##################################################
# Trajectories to Stealthy Ground States
##################################################


class TrajectoryEnsemble(Ensemble):
    """
    Class for storing ensemble of trajectories to stealthy ground states
    and computing basic (pair) statistics
    """

    def __init__(self, members: list = []) -> None:
        return super().__init__(members)

    def __repr__(self) -> None:
        res = ""
        for midx in range(self.n_members):
            _, trajectory = self.members[midx]
            snapshots, n_particles, dimension = trajectory.shape
            res += f"Member {midx:>3d}: {dimension:>2d}D trajectory over {snapshots:>4d} snapshots of {n_particles:>5d} points\n"
        return res

    def _add_member(self, lv: np.array, trajectory: np.array) -> None:
        self.members.append(TrajectoryMember(lv, trajectory))
        self.n_members += 1
        return 0

    def normalize_trajectories(self) -> None:
        """
        Normalize trajectories wrt time
        """
        I = []

        for midx in range(self.n_members):
            ensemble_member = self.members[midx]
            lv = ensemble_member.lv
            trajectory = ensemble_member.trajectory

            n_snapshots, _, _ = trajectory.shape
            I.append(n_snapshots)

        return 0

    def compute_statistic(self, stat: str) -> None:
        """
        Compute ensemble-average statistic over entire trajectory
        Assumes trajectories are of equal size so that averaging is straightforward
        """
        if stat not in self.stat_funcs:
            print(f"Requested statistic '{stat}' not supported")
            sys.exit(1)

        ufunc = self.stat_funcs[stat].ufunc

        eres = [[] for _ in range(self.n_members)]  # ensemble average of this trajectory

        for midx in range(self.n_members):
            print(f"Member {midx:d}...")

            ensemble_member = self.members[midx]
            lv = ensemble_member.lv
            trajectory = ensemble_member.trajectory

            for sidx in range(trajectory.shape[0]):

                coordinates = trajectory[sidx]

                z = ufunc(coordinates, lv)
                eres[midx].append(z)

        # # average over members at each snapshot in time
        # # THIS MUST BE CHECKED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # eres = np.mean(eres, axis = 1)
        # eres = np.transpose(eres, (1, 0, 2, 3))
        # eres = np.mean(eres, axis = 0)

        return 0


if __name__ == "__main__":
    import glob

    CONFIG_DIR = sys.argv[1]

    patterns = []
    for ifile in glob.glob(CONFIG_DIR + "/*__*"):
        _, _, lv, coords = read_cco(ifile)
        patterns.append((lv, coords))

    Omega = PointPatterns(patterns)

    print(Omega)
