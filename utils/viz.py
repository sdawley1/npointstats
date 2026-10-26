# %%
#!/bin/bash

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=E0401cond
sys.path.append("/home/sd2402/n_point_statistics/")
from utils.readwrite import read_cco, read_xd_data

# ===== plotting parameters =====
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.linewidth"] = 2

# ==================================================
# PLOTTING FUNCTIONS
# ==================================================

def plot_g2(g2: np.ndarray, sg: Optional[Dict[str, Any]] = None) -> None:

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    plt.xlabel(r"r/D")
    plt.ylabel(r"$g_2(r)$", fontsize=16)

    plt.plot(*g2.T, **sg)

    plt.show()



if __name__ == "__main__":

    # ===== directories =====
    tmp = Path("/home/sd2402/n_point_statistics/src/cpu_analysis")

    # ===== load data =====
    g2 = read_xd_data(tmp.joinpath("g2.txt"))

    # ===== plotting =====
    plot_g2(g2)

    print("Done")

# %%