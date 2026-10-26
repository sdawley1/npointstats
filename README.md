# npointstats

Structure factor S(k), pair correlation function g₂(r), and local number
variance σ²(R) of d-dimensional point patterns in periodic boxes of **any
shape**, plus liquid-state reference closures (Percus–Yevick, Baus–Colot) and
Lado radial Fourier transforms.

- Correct for triclinic (sheared, rhombic, …) cells, not only cubes.
- S(k) on every reciprocal-lattice vector with 0 < |k| ≤ k_max from one
  FINUFFT transform: O(N + M log M) time, O(N + M) memory.
- g₂(r) and σ²(R) from chunked KD-tree queries: memory stays bounded at any N.
- Reproducible: seeded number-variance windows, sorted file order, and
  parallel runs that match serial runs exactly.
- Tested: ~400 unit, integration, property (Hypothesis) and regression tests,
  97 % line coverage; every defect found in the original code is logged in
  [`DEFECTS.md`](DEFECTS.md) with a regression test (including the ten found
  in an independent review of the new code).

## Installation

Python ≥ 3.10. Dependencies: NumPy, SciPy, FINUFFT (wheels on PyPI).

```bash
# from a clone (development install, with test tools)
pip install -e ".[dev]"

# or into a fresh micromamba env on Della (login node: compute nodes are offline)
micromamba create -n npstats -c conda-forge python=3.12 numpy scipy finufft pip
micromamba activate npstats
export PYTHONNOUSERSITE=1     # keep stale ~/.local packages out
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
import npointstats as nps

rng = np.random.default_rng(0)
pattern = nps.PointPattern(rng.random((2000, 2)), nps.Box.cubic(1.0, 2))

sk = nps.structure_factor(pattern)  # all k with 0 < |k| <= 6 pi rho^(1/2)
sk_binned = sk.radial_average()  # shells of width 2 pi / L
g2 = nps.pair_correlation(pattern, dr=0.002)  # r up to min(10 rho^-1/2, L/2)
nv = nps.number_variance(pattern, n_windows=5000, seed=1)
nv_exact = nps.number_variance_from_sk(sk, nv.radii)  # same quantity via S(k)

# ensembles (members may have different boxes and N)
ens = nps.Ensemble.from_files("patterns/", glob="*__*", fmt="cco")
nps.save_result(ens.structure_factor(jobs=8), "sf.csv")
```

`examples/quickstart.py` runs a longer version end to end.

## Command line

```bash
npointstats -d patterns/ -s sf g2 nv --jobs 8 --seed 1 --out results/
npointstats -d data/ -p simple --box 161.80216 161.80216 -s nv    # plain columns
npointstats -d data/ -p simple --box 50 50 --columns 1 2 -s g2     # skip an id column
python -m npointstats --help
```

The command writes `sf.csv`, `g2.csv` and `nv.csv`, each with a commented
metadata header. `--raw-sk` also writes the unbinned S(k) of each member to
`sk_raw/<file name>.npz`. Each CSV has `#` metadata lines followed by one
header row; read it back with `npointstats.load_table(path)` (data, column
names, metadata) or `pandas.read_csv(path, comment="#")`.
On a cluster, use one process per member (`--jobs`) and one FINUFFT thread
(`--nthreads 1`, the default); see `examples/slurm_job.sh`.

| format (`-p`) | reader | layout |
|---|---|---|
| `cco` | `read_cco` | d; d lattice rows; one point per line (CCOptimization input) |
| `dnv` | `read_dnv` | Donev event-driven MD output (`box::WriteConfiguration`) |
| `lstj` | `read_lstj` | Lubachevsky–Stillinger / Torquato–Jiao output |
| `morsefire` | `read_morsefire` | directory with `boxSize.dat`, `positions.dat`, `scalars.dat` |
| `simple` | `read_simple` | whitespace/CSV columns; box from `--box` |

## Conventions

- **Box.** Lattice vectors are the *rows* of a (d, d) matrix L. A point with
  fractional coordinates s is at x = s·L. Positions are **Cartesian** in every
  reader and writer, and need not lie in the primary cell.
- **Structure factor.** S(k) = |Σⱼ e^{ik·rⱼ}|² / N on reciprocal vectors
  k = n·B, with B = 2π (L⁻¹)ᵀ, so aᵢ·bⱼ = 2πδᵢⱼ. k = 0 (forward scattering)
  is excluded. The full inversion-symmetric set inside |k| ≤ k_max is used,
  so every radial shell is complete. The default is
  k_max = 3·2πρ^{1/d}.
- **Radial bins.** S(k) bins are centred on multiples of dk (edges at
  (j + ½)dk); only bins entirely below k_max are reported. g₂ bins are
  [j·dr, (j+1)·dr). All bins are half-open, [lo, hi).
- **g₂ normalisation.** g₂ = n_pairs·V / (N(N−1)[v(r_{i+1}) − v(r_i)]), with
  exact d-ball shell volumes, so a binomial (uniform random) pattern has
  E[g₂] = 1. r_max cannot exceed the inscribed radius of the cell, which is
  half the smallest face separation of the *reduced* cell. Internally each
  `Box` LLL/Minkowski-reduces its lattice, so this limit, the minimum image,
  and the S(k) wave-vector grid depend only on the lattice, not on the basis
  you supply (e.g. a sheared Lees–Edwards cell).
- **Number variance.** N(R) counts points strictly inside a window of
  radius R centred at a uniform random point. σ²(R) is the sample variance
  (ddof = 1). `number_variance_from_sk` uses the exact torus identity
  σ² = ρv(R)·(1/V)·Σ_{k≠0} S(k) α̃₂(k;R) (Torquato & Stillinger 2003). Vectors
  beyond k_max are included assuming S = 1 there (`ideal_tail=True`).
- **Errors.** For one pattern, `sem` is the within-bin (S, σ²) or Poisson
  counting (g₂) error. For an ensemble it is the member-to-member standard
  error, and `counts` is the number of members contributing to each bin.
- **Lattices.** For crystals, pick bin widths whose edges do not coincide with
  shell radii: a distance exactly on an edge can fall on either side by one
  ulp.
- **Theory subpackage.** c̃(k) = ∫ c(r) e^{−ik·r} dᵈr is returned *without*
  the density factor, so S(k) = 1/(1 − ρc̃(k)) with ρ = η / v₁(D/2). PY is
  available for d = 1 and 3 (d = 2 raises). BC is available for d = 1, 2, 3
  and uses the Z₆ rescaled-virial equations of state in 2D/3D (Tonks in 1D).
  At contact, c(r) takes the midpoint value at r = D.

## Performance notes

- FINUFFT's default (all cores) is extremely slow on shared many-core nodes
  for small and medium problems: a 2D N = 3000 transform took 18 s with 192
  threads and 0.07 s with one. The package therefore defaults to
  `nthreads=1` and parallelises over ensemble members.
- On shared login nodes, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
  Otherwise BLAS/OpenMP start one thread per core (192 on Della login
  nodes) and contend with other users. SLURM jobs are already limited to
  their allocated CPUs.
- g₂ and σ² cost ∝ N·ρ·v(r_max). The defaults (10 and 5 mean spacings) are
  cheap. Asking for r_max ≈ L/2 is inherently O(N²).
- Benchmarks against the original code are in
  [`docs/AUDIT_2026-10-26.md`](docs/AUDIT_2026-10-26.md).

## Migrating from the original scripts

| original | npointstats |
|---|---|
| `python master.py -d DIR -s sf -n 'cco*'` | `npointstats -d DIR -s sf -n 'cco*'` |
| `utils.readwrite.read_cco(f)` → `(d, N, lv, x)` | `npointstats.read(f, "cco")` → `PointPattern` (`.box.lattice`, `.positions`) |
| `utils.point_pattern.Ensemble(members)` | `npointstats.Ensemble(patterns)` or `Ensemble.from_files(dir, glob)` |
| `sample_xd_sf(x, lv)` / `sample_xd_sf_collective(x, lv)` | `structure_factor(PointPattern(x, lv), k_max=...)` |
| `angular_average(Z, dk=...)` | `StructureFactor.radial_average(dk=...)` or `radial_average(x, y, edges)` |
| `sample_xd_g2(x, lv, delta_r)` | `pair_correlation(pattern, dr=..., r_max=...)` (dr is absolute) |
| `sample_xd_nv(x, lv)` (freud) | `number_variance(pattern, radii, n_windows, seed)` |
| `utils/approx/*`, `utils/nft/*` | `npointstats.theory` (`py_*`, `bc_*`, `radial_fourier_transform`) |
| `sf_res.txt`, `sf_bin.txt` | `sf.csv` (binned) and `--raw-sk` → `sk_raw/<file>.npz` |

Numbers change wherever the original code had a defect (see `DEFECTS.md`):
- g₂ at small r rises by the shell-volume correction (2.5 % at contact for
  2D RSA).
- S(k) changes in non-orthogonal boxes and in the large-|k| shells.
- Ensemble averages change when members have different boxes.

The original code is preserved in the first commit (`363ae8f`).

## Development

```bash
pytest                                   # ~400 tests, ~20 s
pytest --hypothesis-profile=thorough     # 2000 examples per property
pytest --cov                             # coverage report
ruff check . && ruff format --check . && mypy
pre-commit install                       # run the fast checks on every commit
```

Tests live in `tests/{unit,property,integration,regression}`. Every defect gets
an entry in `DEFECTS.md` and a `test_regression_D###_*` test that failed
before the fix.

## License

MIT; see [`LICENSE`](LICENSE).
