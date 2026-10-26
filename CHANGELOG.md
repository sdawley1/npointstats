# Changelog

## 0.1.0 — 2026-10-26

First packaged release. Rewrite of the original `n_point_statistics` scripts
(preserved in commit `363ae8f`).

### Added
- `npointstats` package (src layout, `pyproject.toml`, MIT license) with
  `Box`, `PointPattern`, `Ensemble`, `structure_factor`, `pair_correlation`,
  `number_variance`, `number_variance_from_sk`, `radial_average`, readers for
  CCO / Donev / LSTJ / morse-FIRE / plain files, lossless `write_cco`, and
  self-describing CSV/NPZ result files.
- `npointstats` command-line tool (`python -m npointstats`).
- `npointstats.theory`: Lado radial Fourier transforms, Percus–Yevick (d = 1, 3)
  and Baus–Colot (d = 1, 2, 3) direct correlation functions and S(k).
- Test suite: unit, integration, Hypothesis property, and regression tests.
- CI workflow, pre-commit configuration, benchmarks, example SLURM job.

### Fixed
- 46 defects logged in `DEFECTS.md` (D-001 … D-046), among them:
  - wrong S(k) and g₂ in non-orthogonal boxes;
  - a biased g₂ shell volume and bin positions;
  - one-octant wave-vector sampling and incomplete large-|k| shells;
  - the dropped last radial bin;
  - element-wise ensemble averaging across different boxes;
  - O(N²) / O(M·N) memory blow-ups;
  - unseeded, mis-dimensioned number-variance windows;
  - six-decimal CCO output;
  - CLI defaults that read no files;
  - errors in the PY / BC / Lado theory code.

### Removed
- Unused or non-working modules: GPU (numba.cuda) and JAX prototypes,
  trajectory classes, CUDA device query, Perlin noise, scratch `tmp*.py`
  copies, empty notebooks and modules.
