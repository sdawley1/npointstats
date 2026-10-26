# Defect log

## D-001 — sample_xd_sf rescales the caller's coordinates in place
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:214-243 (`sample_xd_sf`)
- **Symptom:** `np.asarray` does not copy a float64 array, so the in-place `/=` maps the caller's positions to [-pi, pi)
- **Correction (2026-10-26):** re-run on 363ae8f shows the in-place step is only `point_pattern[:, k] /= lv[k, k]`; the later `np.add` makes a new array. The caller's array is therefore divided by the box lengths (unchanged for a unit box), not mapped to [-pi, pi).
- **Repro:** `x = np.random.rand(10, 3); sample_xd_sf(x, np.eye(3)); x.min()` → about `-pi`, expected unchanged in [0, 1) (corrected repro: `x = 2*np.random.rand(10, 3); sample_xd_sf(x, 2*np.eye(3))` → `x.max()` ≈ 0.98, was 1.96)
- **Impact:** any later use of `coords` after the call (e.g. a second S(k) or g2 on the same array) sees rescaled positions
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D001_positions_not_mutated (+ property test_regression_D001_no_mutation_property); seen failing on 363ae8f: box 2*eye(3) left x.max() = 0.98 (was 1.96)
- **Fix:** structure_factor works on PointPattern.fractional() copies; PointPattern positions are read-only

## D-002 — angular_average drops the last populated bin
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:120-130 (`angular_average`)
- **Symptom:** the loop runs over `unique.size - 1` bins and the return slices `W[:-1]`, so the bin with the largest |k| is never reported; the slicing `X[ux1:ux2]` also assumes `X` is already sorted by bin
- **Repro:** `angular_average(np.array([[0., 1.], [1., 2.], [2., 3.]]), xidx=0, yidx=1, dk=1.0)` → 1 row `[0, 1]`; the bin holding x = 1, 2 (mean S = 2.5) is lost
- **Impact:** the highest-|k| shell is missing from every binned S(k); unsorted input gives wrong means
- **Regression test:** tests/regression/test_regression_binning.py::test_regression_D002_last_bin_kept_and_unsorted_input (+ property test_regression_D002_counts_and_means_conserved); seen failing on 363ae8f: legacy angular_average returned [[0.0, 1.0]] only
- **Fix:** binning.radial_average uses searchsorted + bincount over explicit edges (order-independent, all bins kept)

## D-003 — __main__ unpacks two values from testing(), which returns one
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:470 (`__main__`)
- **Symptom:** `res, aa_res = testing(...)` unpacks a (W, d+2) array; raises ValueError unless W == 2
- **Repro:** `python c_structure_factor.py <dir> '*'` → `ValueError: too many values to unpack`
- **Impact:** the script cannot be run as documented
- **Regression test:** none yet
- **Fix:** none yet

## D-004 — sample_xd_sf ignores its kn argument
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:219-225 (`sample_xd_sf`)
- **Symptom:** the `match dimension` block overwrites `kn` unconditionally, so a caller-supplied grid has no effect
- **Repro:** `sample_xd_sf(x, np.eye(3), kn=(5, 5, 5)).shape[0]` → `15624` (25^3 - 1), expected `124`
- **Impact:** every 3D call uses a 25^3 grid regardless of the request
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D004_k_max_respected (+ property test_regression_D004_count_matches_enumeration); seen failing on 363ae8f: kn=(5,5,5) returned 15624 vectors
- **Fix:** structure_factor(k_max=...) samples exactly the reciprocal vectors with 0 < |k| <= k_max

## D-005 — Python module has a bash shebang
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:1
- **Symptom:** `#!/bin/bash` on a Python file; executing it directly runs bash on Python source
- **Repro:** `chmod +x c_structure_factor.py && ./c_structure_factor.py` → bash syntax errors
- **Impact:** only direct execution; `python3 c_structure_factor.py` is unaffected
- **Regression test:** none yet
- **Fix:** none yet

## D-006 — Ensemble uses a mutable default argument
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/point_pattern.py:100 (`Ensemble.__init__`)
- **Symptom:** `members: list = []` is shared by every `Ensemble()`; `load_cco` appends to it, so members leak between instances
- **Repro:** `a = Ensemble(); a._add_member(lv, x); Ensemble().n_members` → `0`, but `Ensemble().members` already holds `a`'s member
- **Impact:** any script building two ensembles in one process mixes their members
- **Regression test:** none yet
- **Fix:** none yet

## D-007 — parallel_compute_statistic calls a method that does not exist
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/point_pattern.py:211 (`Ensemble.parallel_compute_statistic`)
- **Symptom:** `self._check_cuda()` is never defined → AttributeError on every call
- **Repro:** `Ensemble([(np.eye(2), np.random.rand(5, 2))]).parallel_compute_statistic("sf")` → AttributeError
- **Impact:** the parallel path has never run
- **Regression test:** none yet
- **Fix:** none yet

## D-008 — g2 results are never written; write errors are swallowed
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/point_pattern.py:254 (`write_statistic`), utils/readwrite.py:352 (`write_xd_data`)
- **Symptom:** for every stat except "sf", `stat_results[stat][1]` is None; `write_xd_data(None)` raises TypeError, the bare `except:` prints "Output unsuccessfully written" and returns False, which nobody checks
- **Repro:** `master.py -s g2 ...` → `g2_res.txt` is written, then "Output unsuccessfully written" for `g2_bin.txt`; exit code 0
- **Impact:** silent partial output; any write failure (disk full, permissions) is hidden
- **Regression test:** none yet
- **Fix:** none yet

## D-009 — ensemble average is element-wise across members with different boxes
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/point_pattern.py:182-183 and :230-231 (`compute_statistic`, `parallel_compute_statistic`)
- **Symptom:** raw per-member S(k) rows are averaged index-by-index, so members whose boxes differ average S at different k vectors; for g2 the bin count depends on the box, so `np.asarray(eres)` becomes ragged and raises
- **Repro:** two members with boxes `np.eye(2)` and `2*np.eye(2)` → averaged row 0 mixes |k|=2π with |k|=π
- **Impact:** wrong ensemble S(k) whenever members are not in identical boxes (e.g. NPT outputs, jammed packings)
- **Regression test:** none yet
- **Fix:** none yet

## D-010 — S(k) bin width hard-coded and inconsistent between serial and parallel paths
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/point_pattern.py:190 (π/100) and :237 (π/20)
- **Symptom:** the radial bin width is a fixed absolute constant, independent of box size and not exposed to the user; the two code paths disagree by 5x
- **Repro:** same ensemble through `compute_statistic` vs `parallel_compute_statistic` → different number of `sf_bin.txt` rows
- **Impact:** binned S(k) resolution silently depends on the code path and on box units
- **Regression test:** none yet
- **Fix:** none yet

## D-011 — collective S(k) samples only non-negative wave-vector indices
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:170 (`generate_wavevectors`, same in src/gpu_analysis/g_structure_factor.py:185)
- **Symptom:** `itertools.product(*map(range, kn))` yields only n_i ≥ 0, i.e. one quadrant/octant of k-space
- **Repro:** 2D, `kn=(3, 3)` → no wave vector with a negative component, e.g. (-2π, 2π) is absent
- **Impact:** S(k) of anisotropic patterns (lattices, sheared or triclinic boxes) is biased; shell populations are a quarter (2D) / eighth (3D) of the full shell
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D011_negative_components_present; seen failing on 363ae8f: generate_wavevectors returned no vector with a negative component
- **Fix:** wavevector_indices enumerates the full inversion-symmetric index set

## D-012 — radially averaged S(k) includes incomplete shells beyond the inscribed k-sphere
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:228-238 and :392-397; utils/point_pattern.py:195
- **Symptom:** wave vectors fill a box in index space, so for |k| larger than the radius of the sphere inscribed in that box only the "corners" are sampled
- **Repro:** 2D, `kn=(250, 250)`, unit box → shells with |k| > 2π·125 hold far fewer vectors than 2π|k|/Δk
- **Impact:** large-|k| tail of every binned S(k) is averaged over a direction-biased subset
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D012_complete_shells (property); seen failing on 363ae8f: shell |k|=300*2pi held 452 vectors, complete shell has 1848
- **Fix:** wave vectors are cut by |k| <= k_max; StructureFactor.radial_average keeps only bins entirely below k_max

## D-013 — collective S(k) needs O(M·N) memory
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:343 (`_compute_sf_all_k`)
- **Symptom:** `jax.vmap` over all M wave vectors materialises the (M, N) phase matrix
- **Repro:** 2D default `kn=(250, 250)` with N = 1e5 → ~6.2e4 × 1e5 × 8 B ≈ 50 GB
- **Impact:** the default S(k) path (used by `Ensemble`) cannot run on the N = 1e5 configurations in CONFIGURATIONS
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D013_direct_sum_memory_bounded; seen failing on 363ae8f: N=2000 2D collective S(k) peaked at 1.64 GB RSS (M*N = 62499*2000), i.e. ~80 GB at N=1e5
- **Fix:** default method is one FINUFFT type-1 transform (O(N + M) memory); the direct reference sums in chunks of 2^22 phases

## D-014 — finufft S(k) rescales coordinates by diagonal box entries only
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:241-243 (`sample_xd_sf`)
- **Symptom:** positions are divided by `lv[k, k]`, which is only the fractional coordinate for diagonal boxes; wave vectors use the full reciprocal lattice
- **Repro:** triangular lattice in its rhombic primitive box → Bragg peaks appear at the wrong k
- **Impact:** wrong S(k) for every non-orthogonal box
- **Regression test:** tests/regression/test_regression_structure_factor.py::test_regression_D014_triclinic_bragg_peaks; seen failing on 363ae8f: triangular crystal (m=6) in rhombic cell had its S=36 peaks at |k| = 267-487 instead of 7.255
- **Fix:** FINUFFT runs on 2*pi*fractional coordinates, exact for any cell

## D-015 — minimum image uses lattice-vector norms (wrong for non-orthogonal boxes)
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:133-213 (`dist_matrix_1d/2d/3d`); utils/geom.py:115-151 (`nearest_image`)
- **Symptom:** each Cartesian component is wrapped by `|a_i|`, which is the period along axis i only when the box is axis-aligned and orthogonal
- **Repro:** box `[[1, 0], [0.5, 0.866]]`, points (0, 0) and (0.5, 0.866) (a lattice translate, true distance 0) → reported distance ≈ 0.518
- **Impact:** wrong g2 for triangular, sheared, or triclinic cells
- **Regression test:** tests/regression/test_regression_geometry.py::test_regression_D015_triclinic_minimum_image (+ property test_regression_D015_lattice_translates_vanish); seen failing on 363ae8f: legacy dist_matrix_2d gives 0.5176 for a lattice translate
- **Fix:** npointstats.box.Box.minimum_image rounds fractional coordinates; exact below Box.inscribed_radius for any cell shape

## D-016 — nearest_image mutates its argument in place
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/geom.py:132-136 (`nearest_image`, p2 is None branch)
- **Symptom:** `p1[i] -= k` edits the caller's array
- **Repro:** `p = np.array([1.5, 0.2]); nearest_image(np.eye(2), p); p` → `[0.5, 0.2]`
- **Impact:** callers (e.g. CONFIGURATIONS/RSA/rsa.py imports a copy of this function) see silently moved points
- **Regression test:** tests/regression/test_regression_geometry.py::test_regression_D016_wrap_does_not_mutate (+ property test_regression_D016_no_mutation_property); seen failing on 363ae8f: legacy nearest_image left input as [0.5, 0.2]
- **Fix:** Box.wrap/minimum_image/to_fractional return new arrays; PointPattern stores a read-only copy

## D-017 — sample_xd_g2 silently divides delta_r by the box length
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_npt_correlation.py:241
- **Symptom:** `delta_r /= vol ** (1 / dimension)` makes the bin width relative to L, contrary to the docstring ("spacing between successive radii")
- **Repro:** box `10*np.eye(2)`, `delta_r=0.1` → bins of width 0.01
- **Impact:** g2 resolution is wrong for any box with L ≠ 1
- **Regression test:** none yet
- **Fix:** none yet

## D-018 — sample_xd_g2 ignores max_l and uses np.max(lv)/2 as range
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_npt_correlation.py:243-246
- **Symptom:** `max_l` is computed but unused; the bins always run to half the largest *matrix entry* of `lv`, which can exceed the minimum-image limit (half the smallest box width)
- **Repro:** `sample_xd_g2(x, np.diag([1., 4.]), max_l=0.1)` → bins to r = 2.0, beyond the valid r < 0.5
- **Impact:** g2 beyond half the shortest box width is corrupted by double counting / missed images
- **Regression test:** none yet
- **Fix:** none yet

## D-019 — g2 shell volume centred on the upper bin edge
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_npt_correlation.py:264-266 (also src/gpu_analysis/g_npt_correlation.py:54)
- **Symptom:** histogram bin [r_i, r_{i+1}) is normalised by a shell of width dr centred at r_{i+1}, and reported at r_{i+1}; correct is v(r_{i+1}) - v(r_i) at the bin centre
- **Repro:** 2D Poisson, dr = 0.05, first bin [0.05, 0.10) → g2 ≈ 0.75 instead of 1 (old shell π(0.125²−0.075²) = 0.0314 vs exact π(0.10²−0.05²) = 0.0236)
- **Impact:** O(dr/r) bias at small r; peaks shifted by dr/2
- **Regression test:** none yet
- **Fix:** none yet

## D-020 — sample_xd_g2 builds a dense N×N distance matrix
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_npt_correlation.py:254-262
- **Symptom:** O(N²) memory and time
- **Repro:** N = 1e5 (CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_100000*) → 80 GB
- **Impact:** g2 is unusable on the group's large configurations
- **Regression test:** none yet
- **Fix:** none yet

## D-021 — importing c_npt_correlation forces JAX onto the CPU process-wide
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_npt_correlation.py:21-22
- **Symptom:** sets `JAX_PLATFORMS=cpu` and `JAX_SKIP_CUDA_CONSTRAINTS_CHECK` in `os.environ` at import time
- **Repro:** `import utils.point_pattern` then any JAX GPU code in the same process runs on CPU
- **Impact:** hidden global side effect on unrelated code
- **Regression test:** none yet
- **Fix:** none yet

## D-022 — number variance windows get a random z-coordinate in 2D
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** src/cpu_analysis/c_number_variance.py:114 (`sample_xd_nv`)
- **Symptom:** `np.random.rand(M, 3)` gives non-zero z for 2D boxes, which freud rejects (or mis-wraps); RNG is unseeded; neighbour counts accumulate in a Python loop over bonds (line 124)
- **Repro:** any 2D pattern → freud error / wrong counts; two runs give different σ²(R)
- **Impact:** 2D number variance wrong or crashes; results not reproducible
- **Regression test:** none yet
- **Fix:** none yet

## D-023 — CLI default file glob matches nothing
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** master.py:139-141 and :176 (`parse_input`, `--name` default `"__"`)
- **Symptom:** default `"__"` becomes glob `"__*"`, matching only names *starting* with `__`; files like `RSA..__0000.txt` are skipped; an empty ensemble then averages to NaN
- **Repro:** `python master.py -d patterns -s sf` on a CCO directory → "Mean of empty slice" warning, NaN output
- **Impact:** default invocation silently produces nothing useful
- **Regression test:** none yet
- **Fix:** none yet

## D-024 — CLI: -s optional though required, -f unused, unsorted file order
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** master.py:162-164 (`-s`), :154-160 (`-f/--hdf5`), :98 (`dp.glob`)
- **Symptom:** omitting `-s` → `TypeError: 'NoneType' object is not iterable`; `--hdf5` is parsed and never used; `Path.glob` order is filesystem-dependent, so member order (and parallel output order) is not reproducible
- **Repro:** `python master.py -d patterns` → TypeError traceback
- **Impact:** confusing CLI, non-reproducible member ordering
- **Regression test:** none yet
- **Fix:** none yet

## D-025 — read_simple_data / read_chiral_active hard-code the box
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** master.py:51 (`read_simple_data` → always 2×2 identity), master.py:69 (`read_chiral_active` → 161.80216 square)
- **Symptom:** box ignores the data's dimension and size
- **Repro:** a 3D "simple" file → (3,) positions paired with a (2,2) box → crash or wrong S(k)
- **Impact:** wrong statistics for any file not matching the hard-coded box
- **Regression test:** tests/regression/test_regression_io.py::test_regression_D025_simple_reader_uses_given_box; seen failing on 363ae8f: legacy read_simple_data returned a 2x2 identity box for 3D data
- **Fix:** io.read_simple requires the box; chiral-active files are read with read_simple(columns=2, box=...)

## D-026 — read_cco breaks on trailing blank lines
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/readwrite.py:94-102 (`read_cco`, and its copies in the cpu/gpu modules)
- **Symptom:** blank lines are counted as particles and produce empty coordinate rows → ragged `np.array` (ValueError in NumPy ≥ 1.24)
- **Repro:** a valid CCO file with a trailing `"\n"` line → ValueError
- **Impact:** files edited by hand or concatenated fail to read; `n_particles` is wrong
- **Regression test:** tests/regression/test_regression_io.py::test_regression_D026_cco_trailing_blank_lines; seen failing on 363ae8f: legacy read_cco raised ValueError (inhomogeneous array)
- **Fix:** io.read_cco ignores blank lines and reports the offending line number on parse errors

## D-027 — write_cco truncates coordinates to 6 decimals
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/readwrite.py:35 (`stringify`, also utils/geom.py:233)
- **Symptom:** `f"{x:0.6f}"` loses all digits below 1e-6
- **Repro:** `write_cco(np.eye(2), np.array([[0.1234567891, 0.5]]), f)` then `read_cco(f)` → 0.123457
- **Impact:** round trip is lossy; jammed/contact configurations written with it overlap or separate by ~1e-6
- **Regression test:** tests/regression/test_regression_io.py::test_regression_D027_cco_full_precision (+ property test_regression_D027_round_trip_property); seen failing on 363ae8f: legacy round trip gave 0.123457
- **Fix:** io.write_cco writes %.17g (bit-exact round trip)

## D-028 — read_morsefire is 2D-only and fails on missing files
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/readwrite.py:253-272 (`read_morsefire`)
- **Symptom:** only `boxsize[0]`, `boxsize[1]` are placed in the box (3D boxes get a zero row → singular); if any of the four files is absent, `UnboundLocalError`; `glob("*")` substring matching also picks up e.g. `old_positions.dat`
- **Repro:** 3D morse output → zero-volume box; directory without `radii.dat` → UnboundLocalError; under NumPy 2 any `boxSize.dat` (one value per line or all on one line) → `ValueError: setting an array element with a sequence`
- **Impact:** unusable for 3D, cryptic errors otherwise
- **Regression test:** tests/regression/test_regression_io.py::test_regression_D028_morsefire_3d, ::test_regression_D028_morsefire_missing_file; seen failing on 363ae8f: legacy read_morsefire raised 'ValueError: setting an array element with a sequence' (NumPy 2) for boxSize.dat with one value per line
- **Fix:** io.read_morsefire reads d side lengths, matches exact file names, raises FileNotFoundError naming missing files

## D-029 — library code calls sys.exit
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/readwrite.py:139, :185; utils/point_pattern.py:79, :165, :215; src/cpu_analysis/c_structure_factor.py:433
- **Symptom:** invalid input terminates the interpreter instead of raising an exception
- **Repro:** `Ensemble().compute_statistic("xyz")` → process exits with status 1 (kills Jupyter kernels)
- **Impact:** callers cannot handle errors
- **Regression test:** none yet
- **Fix:** none yet

## D-030 — modules depend on a non-existent sys.path entry
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** master.py:20, utils/point_pattern.py:25, src/cpu_analysis/c_npt_correlation.py:30 (and other modules)
- **Symptom:** `sys.path.append("/home/sd2402/n_point_statistics/")` — the directory does not exist; imports only work when run from the repo root
- **Repro:** `cd /tmp && python /path/to/master.py` → ModuleNotFoundError: utils
- **Impact:** the code cannot be installed or shared
- **Regression test:** none yet
- **Fix:** none yet

## D-031 — every Python module has a bash shebang
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** first line of every .py file except utils/perlin_noise.py (generalises D-005)
- **Symptom:** `#!/bin/bash` on Python source
- **Repro:** `chmod +x master.py && ./master.py` → bash syntax errors
- **Impact:** direct execution fails
- **Regression test:** none yet
- **Fix:** none yet

## D-032 — utils.geom requires JAX at import for trivial helpers
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/geom.py:14-17
- **Symptom:** `import utils.geom` (needed by point_pattern for `angular_average`) imports jax and toggles x64 globally
- **Repro:** env without jax → `import utils.point_pattern` fails
- **Impact:** heavy, side-effecting dependency for NumPy-only code
- **Regression test:** none yet
- **Fix:** none yet

## D-033 — trapz_integrate implements neither the trapezoid nor Simpson's rule
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/geom.py:244-263 (and src/gpu_analysis/g_structure_factor.py:101-140)
- **Symptom:** `h[0] = 0` and the weights h_i·f_i·{1,2,4}/3 do not form a quadrature rule
- **Repro:** `trapz_integrate(jnp.linspace(0, 1, 11), jnp.ones(11))` → ≠ 1
- **Impact:** any integral (e.g. σ² from S(k)) computed with it is wrong
- **Regression test:** none yet
- **Fix:** none yet

## D-034 — rho_from_pf takes a square / cube root
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/baus_colot.py:119, :121 (`rho_from_pf`)
- **Symptom:** returns √(4η/πD²) (2D) and ∛(6η/πD³) (3D); ρ = 4η/(πD²), 6η/(πD³)
- **Repro:** `rho_from_pf(0.5, 3, 1.0)` → 0.985, expected 0.955
- **Impact:** the BC density consistency check (and any S(k) built from this ρ) is wrong
- **Regression test:** none yet
- **Fix:** none yet

## D-035 — bc_solve_a passes wrong Jacobians to fsolve
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/baus_colot.py:281 (2D, missing 2η² factor) and :294 (3D, `2*x` instead of 18η²x)
- **Symptom:** `fprime` is not the derivative of the residual
- **Repro:** compare `fprime(x)` with a finite difference of the residual at η = 0.4 → mismatch
- **Impact:** fsolve may converge slowly or to a wrong root
- **Regression test:** none yet
- **Fix:** none yet

## D-036 — d(ηZ)/dη by finite differences off-grid, past η = 1
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/baus_colot.py:139-140, :442-443
- **Symptom:** the derivative is taken on a 200/500-point grid that does not contain η exactly, and for η > 0.8 the grid runs past η = 1 (pole of Z)
- **Repro:** `bc_dcf(3, 0.45, ...)` → derivative evaluated at η ≈ 0.4498
- **Impact:** BC c(r) inaccurate; Z is analytic, so the derivative should be too
- **Regression test:** none yet
- **Fix:** none yet

## D-037 — Lado transform sums drop the last node
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/nft/numerical_fourier_transform.py:60, :86, :108, :133, :156, :170
- **Symptom:** `range(1, N - 1)` instead of `range(1, N)`; Lado's sums run over i = 1…N−1
- **Repro:** forward then inverse transform of a Gaussian does not return the input to machine precision
- **Impact:** loss of the discrete orthogonality that makes the Lado scheme exact
- **Regression test:** none yet
- **Fix:** none yet

## D-038 — percus_yevick.py cannot be imported or called
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/percus_yevick.py:17-19 (bad path, `nft_1d` not exported), :125/:127 (`D=` passed to functions without `D`), :45 (array passed where callable expected); d = 2 returns None
- **Symptom:** ImportError on import; TypeError if fixed
- **Repro:** `import utils.approx.percus_yevick` → ImportError
- **Impact:** PY reference S(k) unavailable
- **Regression test:** none yet
- **Fix:** none yet

## D-039 — PY and BC disagree on the ρ convention of c̃(k)
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/baus_colot.py (`bc_dcf_*` returns ρc̃), utils/approx/percus_yevick.py (`py_dcf_*` returns c̃)
- **Symptom:** S(k) = 1/(1 − ρc̃) must be formed differently for each; tmp scripts multiply ρ twice
- **Repro:** see utils/nft/tmp-2.py:510, :519
- **Impact:** comparisons between closures are wrong by a factor ρ
- **Regression test:** none yet
- **Fix:** none yet

## D-040 — Approx ABC makes its dispatcher abstract
- **Found:** 2026-10-26, during the repository audit
- **Status:** open
- **Where:** utils/approx/approx_class.py (`apx_dcf` abstract; returns `1` for unsupported d)
- **Symptom:** subclasses must re-implement the dispatcher; unsupported d returns 1 instead of raising
- **Repro:** n/a (no subclasses exist)
- **Impact:** design defect only
- **Regression test:** none yet
- **Fix:** none yet

## D-041 — removed scratch modules (logged for the record)
- **Found:** 2026-10-26, during the repository audit
- **Status:** wontfix (modules removed in the v0.1 refactor; recoverable from the snapshot commit)
- **Where / Symptom:**
  - src/gpu_analysis/g_pairwise_distances.py:118-120 — 3D box length from `hypot` of the first two components → cz = 0, z never wrapped; :55 `fill_diagonal` corrupts cross-distance matrices; atomics where none are needed
  - src/gpu_analysis/g_npt_correlation.py:43 overwrites `delta_r`; :125 calls undefined `fft_1d_g2`
  - src/gpu_analysis/g_number_variance.py:19 imports `fsiv`, which is commented out in utils/geom.py → ImportError; jax-jitted `trapz_integrate` called inside `njit`
  - src/gpu_analysis/g_structure_factor.py:24 loads a hard-coded `/home/sd2402/stats/utils/lib_spheres.so` at import
  - src/jax_analysis/*.py — `.at[].set()` results discarded; accumulators never reset per k; `math.cos` on tracers
  - utils/trajectories.py — imports `sample_2d_g3` (commented out), uses undefined `self.stat_funcs`, abstract class never completed
  - utils/cumanagement.py:243 — `int * "unknown"` for GPUs newer than CC 8.6
  - utils/perlin_noise.py:28-31 — `astype(int)` truncates toward zero for negative coordinates
  - utils/viz.py:31 — `**sg` with default None → TypeError
  - src/cpu_analysis/omp_testing.py:18 — midpoint rule off by one (`(j-1) - 0.5`)
  - utils/nft/tmp*.py — stale copies with additional errors (`r = 1/q` grid, ρ counted twice)
- **Impact:** none for users after removal
- **Regression test:** n/a
- **Fix:** removed
