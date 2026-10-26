#!/usr/bin/env python3

"""
Liquid-state reference theory for hard rods, disks, and spheres.

Closed-form and quasi-closed-form direct correlation functions (DCFs) and
structure factors of equilibrium hard-particle fluids, plus Lado's discrete
radial Fourier transforms in d = 1, 2, 3.

Conventions (shared by every function in this subpackage):

- Particles have diameter D (`diameter`, default 1) and packing fraction η;
  the number density is ρ = η / v_1(D/2) (`number_density`).
- Fourier transforms follow F(k) = ∫ f(r) exp(-i k·r) d^d r and
  f(r) = (2π)^{-d} ∫ F(k) exp(i k·r) d^d k.
- c̃(k) is returned WITHOUT a density factor; S(k) = 1 / (1 - ρ c̃(k)).
- Each closure is a set of plain functions taking the dimension `d`; an
  unsupported `d` raises ValueError, and a closure that does not exist in a
  supported dimension (Percus–Yevick in d = 2) raises NotImplementedError.

Modules
-------
hankel
    Lado (1971) radial transforms and the exact transform of ball polynomials.
percus_yevick
    Percus–Yevick DCF, c̃(k), and S(k) for d = 1 (exact) and d = 3.
baus_colot
    Baus–Colot (1987) equation of state, DCF, c̃(k), and S(k) for d = 1, 2, 3.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- local modules -----
from npointstats.theory.baus_colot import (
    bc_compressibility_factor,
    bc_dcf,
    bc_dcf_fourier,
    bc_inverse_compressibility,
    bc_solve_a,
    bc_structure_factor,
    number_density,
)
from npointstats.theory.hankel import (
    ball_polynomial_transform,
    inverse_radial_fourier_transform,
    lado_grid,
    radial_fourier_transform,
)
from npointstats.theory.percus_yevick import (
    py_dcf,
    py_dcf_fourier,
    py_structure_factor,
)

# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "ball_polynomial_transform",
    "bc_compressibility_factor",
    "bc_dcf",
    "bc_dcf_fourier",
    "bc_inverse_compressibility",
    "bc_solve_a",
    "bc_structure_factor",
    "inverse_radial_fourier_transform",
    "lado_grid",
    "number_density",
    "py_dcf",
    "py_dcf_fourier",
    "py_structure_factor",
    "radial_fourier_transform",
]
