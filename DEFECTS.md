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
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:470 (`__main__`)
- **Symptom:** `res, aa_res = testing(...)` unpacks a (W, d+2) array; raises ValueError unless W == 2
- **Repro:** `python c_structure_factor.py <dir> '*'` → `ValueError: too many values to unpack`
- **Impact:** the script cannot be run as documented
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D003_documented_invocation_runs; seen failing on 363ae8f: 'python c_structure_factor.py <dir> *' -> ValueError: too many values to unpack (expected 2)
- **Fix:** script replaced by the npointstats CLI / python -m npointstats

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
- **Status:** fixed
- **Where:** src/cpu_analysis/c_structure_factor.py:1
- **Symptom:** `#!/bin/bash` on a Python file; executing it directly runs bash on Python source
- **Repro:** `chmod +x c_structure_factor.py && ./c_structure_factor.py` → bash syntax errors
- **Impact:** only direct execution; `python3 c_structure_factor.py` is unaffected
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D031_python_shebangs; seen failing on 363ae8f: first line '#!/bin/bash'
- **Fix:** all modules start with #!/usr/bin/env python3

## D-006 — Ensemble uses a mutable default argument
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/point_pattern.py:100 (`Ensemble.__init__`)
- **Symptom:** `members: list = []` is shared by every `Ensemble()`; `load_cco` appends to it, so members leak between instances
- **Repro:** `a = Ensemble(); a._add_member(lv, x); Ensemble().n_members` → `0`, but `Ensemble().members` already holds `a`'s member
- **Impact:** any script building two ensembles in one process mixes their members
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D006_no_shared_member_state; seen failing on 363ae8f: a fresh Ensemble() already held 1 member
- **Fix:** Ensemble stores an immutable tuple given at construction; no mutable defaults

## D-007 — parallel_compute_statistic calls a method that does not exist
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/point_pattern.py:211 (`Ensemble.parallel_compute_statistic`)
- **Symptom:** `self._check_cuda()` is never defined → AttributeError on every call
- **Repro:** `Ensemble([(np.eye(2), np.random.rand(5, 2))]).parallel_compute_statistic("sf")` → AttributeError
- **Impact:** the parallel path has never run
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D007_parallel_path_runs_and_matches; seen failing on 363ae8f: AttributeError: 'Ensemble' object has no attribute '_check_cuda'
- **Fix:** jobs=N uses a ProcessPoolExecutor with ordered map; identical to jobs=1

## D-008 — g2 results are never written; write errors are swallowed
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/point_pattern.py:254 (`write_statistic`), utils/readwrite.py:352 (`write_xd_data`)
- **Symptom:** for every stat except "sf", `stat_results[stat][1]` is None; `write_xd_data(None)` raises TypeError, the bare `except:` prints "Output unsuccessfully written" and returns False, which nobody checks
- **Repro:** `master.py -s g2 ...` → `g2_res.txt` is written, then "Output unsuccessfully written" for `g2_bin.txt`; exit code 0
- **Impact:** silent partial output; any write failure (disk full, permissions) is hidden
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D008_g2_written_and_errors_raised; seen failing on 363ae8f: g2_bin.txt left header-only, 'Output unsuccessfully written', exit 0
- **Fix:** results.save_result writes every statistic; no bare except; CLI exits 1 on any OSError/ValueError

## D-009 — ensemble average is element-wise across members with different boxes
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/point_pattern.py:182-183 and :230-231 (`compute_statistic`, `parallel_compute_statistic`)
- **Symptom:** raw per-member S(k) rows are averaged index-by-index, so members whose boxes differ average S at different k vectors; for g2 the bin count depends on the box, so `np.asarray(eres)` becomes ragged and raises
- **Repro:** two members with boxes `np.eye(2)` and `2*np.eye(2)` → averaged row 0 mixes |k|=2π with |k|=π
- **Impact:** wrong ensemble S(k) whenever members are not in identical boxes (e.g. NPT outputs, jammed packings)
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D009_average_is_binwise (property); seen failing on 363ae8f: row 0 averaged |k|=6.283 with |k|=3.142
- **Fix:** each member is radially binned on common edges before averaging (Ensemble.structure_factor)

## D-010 — S(k) bin width hard-coded and inconsistent between serial and parallel paths
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/point_pattern.py:190 (π/100) and :237 (π/20)
- **Symptom:** the radial bin width is a fixed absolute constant, independent of box size and not exposed to the user; the two code paths disagree by 5x
- **Repro:** same ensemble through `compute_statistic` vs `parallel_compute_statistic` → different number of `sf_bin.txt` rows
- **Impact:** binned S(k) resolution silently depends on the code path and on box units
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D010_bin_width_is_a_parameter; seen failing on 363ae8f: serial pi/100 vs parallel pi/20 hard-coded
- **Fix:** dk is a parameter (default: largest member reciprocal spacing), shared by all code paths

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
- **Regression test:** tests/regression/test_regression_geometry.py::test_regression_D015_triclinic_minimum_image (+ property test_regression_D015_lattice_translates_vanish), tests/regression/test_regression_pair_correlation.py::test_regression_D015_g2_triclinic_translate; seen failing on 363ae8f: legacy dist_matrix_2d gives 0.5176 for a lattice translate
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
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:241
- **Symptom:** `delta_r /= vol ** (1 / dimension)` makes the bin width relative to L, contrary to the docstring ("spacing between successive radii")
- **Repro:** box `10*np.eye(2)`, `delta_r=0.1` → bins of width 0.01
- **Impact:** g2 resolution is wrong for any box with L ≠ 1
- **Regression test:** tests/regression/test_regression_pair_correlation.py::test_regression_D017_dr_is_absolute (property, @example(10.0, 0.1)); seen failing on 363ae8f: dr=0.1 in a 10x10 box gave bins of width 0.01
- **Fix:** pair_correlation(dr=...) uses dr in position units

## D-018 — sample_xd_g2 ignores max_l and uses np.max(lv)/2 as range
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:243-246
- **Symptom:** `max_l` is computed but unused; the bins always run to half the largest *matrix entry* of `lv`, which can exceed the minimum-image limit (half the smallest box width)
- **Repro:** `sample_xd_g2(x, np.diag([1., 4.]), max_l=0.1)` → bins to r = 2.0, beyond the valid r < 0.5
- **Impact:** g2 beyond half the shortest box width is corrupted by double counting / missed images
- **Regression test:** tests/regression/test_regression_pair_correlation.py::test_regression_D018_r_max_respected; seen failing on 363ae8f: max_l=0.1 in a 1x4 box ran to r=1.995
- **Fix:** r_max honoured; default min(10 rho^(-1/d), inscribed radius); larger r_max raises ValueError

## D-019 — g2 shell volume centred on the upper bin edge
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:264-266 (also src/gpu_analysis/g_npt_correlation.py:54)
- **Symptom:** histogram bin [r_i, r_{i+1}) is normalised by a shell of width dr centred at r_{i+1}, and reported at r_{i+1}; correct is v(r_{i+1}) - v(r_i) at the bin centre
- **Repro:** 2D Poisson, dr = 0.05, first bin [0.05, 0.10) → g2 ≈ 0.75 instead of 1 (old shell π(0.125²−0.075²) = 0.0314 vs exact π(0.10²−0.05²) = 0.0236)
- **Impact:** O(dr/r) bias at small r; peaks shifted by dr/2
- **Regression test:** tests/regression/test_regression_pair_correlation.py::test_regression_D019_exact_shell_normalisation (+ property test_regression_D019_single_pair_property); seen failing on 363ae8f: 2D Poisson first bins g2 = 0.751, 0.833, 0.871 reported at bin upper edges
- **Fix:** shell volume v(r_{i+1}) - v(r_i), values reported at bin centres, binomial normalisation N(N-1)/V

## D-020 — sample_xd_g2 builds a dense N×N distance matrix
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:254-262
- **Symptom:** O(N²) memory and time
- **Repro:** N = 1e5 (CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_100000*) → 80 GB
- **Impact:** g2 is unusable on the group's large configurations
- **Regression test:** tests/regression/test_regression_pair_correlation.py::test_regression_D020_memory_bounded; seen failing on 363ae8f: N=15000 peaked at 1.93 GB RSS (dense N^2), 80 GB extrapolated at N=1e5
- **Fix:** chunked KD-tree sparse distances; memory ~ 2^22 pairs per chunk, work ~ N rho v(r_max)

## D-021 — importing c_npt_correlation forces JAX onto the CPU process-wide
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_npt_correlation.py:21-22
- **Symptom:** sets `JAX_PLATFORMS=cpu` and `JAX_SKIP_CUDA_CONSTRAINTS_CHECK` in `os.environ` at import time
- **Repro:** `import utils.point_pattern` then any JAX GPU code in the same process runs on CPU
- **Impact:** hidden global side effect on unrelated code
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D021_D032_no_jax_and_no_environment_side_effects; seen failing on 363ae8f: importing utils.point_pattern set JAX_PLATFORMS=cpu
- **Fix:** no JAX dependency; package import has no side effects

## D-022 — number variance windows get a random z-coordinate in 2D
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** src/cpu_analysis/c_number_variance.py:114 (`sample_xd_nv`)
- **Symptom:** `np.random.rand(M, 3)` gives non-zero z for 2D boxes, which freud rejects (or mis-wraps); RNG is unseeded; neighbour counts accumulate in a Python loop over bonds (line 124)
- **Repro:** any 2D pattern → freud error / wrong counts; two runs give different σ²(R)
- **Impact:** 2D number variance wrong or crashes; results not reproducible
- **Regression test:** tests/regression/test_regression_number_variance.py::test_regression_D022_two_dimensional_windows (+ property test_regression_D022_single_point_window_property); seen failing on 363ae8f: two identical 2D calls differed, sigma^2/v = 2298, 2163 vs binomial 1980, 1937
- **Fix:** number_variance: seeded numpy Generator, windows in the box's own dimension, vectorised KD-tree counts (no freud)

## D-023 — CLI default file glob matches nothing
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** master.py:139-141 and :176 (`parse_input`, `--name` default `"__"`)
- **Symptom:** default `"__"` becomes glob `"__*"`, matching only names *starting* with `__`; files like `RSA..__0000.txt` are skipped; an empty ensemble then averages to NaN
- **Repro:** `python master.py -d patterns -s sf` on a CCO directory → "Mean of empty slice" warning, NaN output
- **Impact:** default invocation silently produces nothing useful
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D023_default_glob_reads_group_file_names; seen failing on 363ae8f: master.py default '-n __' matched nothing, 'Mean of empty slice' warning, exit 0
- **Fix:** default glob '*' (regular, non-hidden files); empty match raises FileNotFoundError -> exit 1

## D-024 — CLI: -s optional though required, -f unused, unsorted file order
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** master.py:162-164 (`-s`), :154-160 (`-f/--hdf5`), :98 (`dp.glob`)
- **Symptom:** omitting `-s` → `TypeError: 'NoneType' object is not iterable`; `--hdf5` is parsed and never used; `Path.glob` order is filesystem-dependent, so member order (and parallel output order) is not reproducible
- **Repro:** `python master.py -d patterns` → TypeError traceback
- **Impact:** confusing CLI, non-reproducible member ordering
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D024_cli_contract; seen failing on 363ae8f: master.py without -s -> TypeError: 'NoneType' object is not iterable
- **Fix:** -s required (argparse exit 2), --hdf5 removed, files read in sorted order

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
- **Status:** fixed
- **Where:** utils/readwrite.py:139, :185; utils/point_pattern.py:79, :165, :215; src/cpu_analysis/c_structure_factor.py:433
- **Symptom:** invalid input terminates the interpreter instead of raising an exception
- **Repro:** `Ensemble().compute_statistic("xyz")` → process exits with status 1 (kills Jupyter kernels)
- **Impact:** callers cannot handle errors
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D029_errors_raise_instead_of_exiting; seen failing on 363ae8f: compute_statistic('xyz') terminated the interpreter with exit code 1
- **Fix:** library raises ValueError/FileNotFoundError; only cli.main converts errors to exit codes

## D-030 — modules depend on a non-existent sys.path entry
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** master.py:20, utils/point_pattern.py:25, src/cpu_analysis/c_npt_correlation.py:30 (and other modules)
- **Symptom:** `sys.path.append("/home/sd2402/n_point_statistics/")` — the directory does not exist; imports only work when run from the repo root
- **Repro:** `cd /tmp && python /path/to/master.py` → ModuleNotFoundError: utils
- **Impact:** the code cannot be installed or shared
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D030_importable_from_any_directory; seen failing on 363ae8f: sys.path entry /home/sd2402/n_point_statistics does not exist (imports only worked from the repo root)
- **Fix:** pip-installable src-layout package; no sys.path manipulation

## D-031 — every Python module has a bash shebang
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** first line of every .py file except utils/perlin_noise.py (generalises D-005)
- **Symptom:** `#!/bin/bash` on Python source
- **Repro:** `chmod +x master.py && ./master.py` → bash syntax errors
- **Impact:** direct execution fails
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D031_python_shebangs; seen failing on 363ae8f: 19 modules started with '#!/bin/bash'
- **Fix:** all modules start with #!/usr/bin/env python3

## D-032 — utils.geom requires JAX at import for trivial helpers
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/geom.py:14-17
- **Symptom:** `import utils.geom` (needed by point_pattern for `angular_average`) imports jax and toggles x64 globally
- **Repro:** env without jax → `import utils.point_pattern` fails
- **Impact:** heavy, side-effecting dependency for NumPy-only code
- **Regression test:** tests/regression/test_regression_ensemble_cli.py::test_regression_D021_D032_no_jax_and_no_environment_side_effects; seen failing on 363ae8f: 'jax' in sys.modules after importing utils.point_pattern
- **Fix:** NumPy/SciPy/FINUFFT only; JAX not a dependency

## D-033 — trapz_integrate implements neither the trapezoid nor Simpson's rule
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/geom.py:244-263 (and src/gpu_analysis/g_structure_factor.py:101-140)
- **Symptom:** `h[0] = 0` and the weights h_i·f_i·{1,2,4}/3 do not form a quadrature rule
- **Repro:** `trapz_integrate(jnp.linspace(0, 1, 11), jnp.ones(11))` → ≠ 1
- **Impact:** any integral (e.g. σ² from S(k)) computed with it is wrong
- **Regression test:** n/a (function removed); seen failing on 363ae8f by inspection: h[0]=0 weights
- **Fix:** removed with utils/geom.py; no custom quadrature remains (theory uses SciPy, sigma^2 from S(k) is a discrete sum)

## D-034 — rho_from_pf takes a square / cube root
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:119, :121 (`rho_from_pf`)
- **Symptom:** returns √(4η/πD²) (2D) and ∛(6η/πD³) (3D); ρ = 4η/(πD²), 6η/(πD³)
- **Repro:** `rho_from_pf(0.5, 3, 1.0)` → 0.985, expected 0.955
- **Impact:** the BC density consistency check (and any S(k) built from this ρ) is wrong
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D034_number_density_no_root; seen failing on 363ae8f: d=2 0.7979 vs 0.6366, d=3 0.9847 vs 0.9549
- **Fix:** theory.baus_colot.number_density: rho = eta / v1(D/2)

## D-035 — bc_solve_a passes wrong Jacobians to fsolve
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:281 (2D, missing 2η² factor) and :294 (3D, `2*x` instead of 18η²x)
- **Symptom:** `fprime` is not the derivative of the residual
- **Repro:** compare `fprime(x)` with a finite difference of the residual at η = 0.4 → mismatch
- **Impact:** fsolve may converge slowly or to a wrong root
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D035_a_equation_jacobian; seen failing on 363ae8f: d=2 fprime(1.5) = -5.614 vs finite difference -1.796; d=3 3.0 vs 4.32
- **Fix:** 3D a in closed form; 2D safeguarded Newton with analytic derivative, bracket [1, 3]

## D-036 — d(ηZ)/dη by finite differences off-grid, past η = 1
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:139-140, :442-443
- **Symptom:** the derivative is taken on a 200/500-point grid that does not contain η exactly, and for η > 0.8 the grid runs past η = 1 (pole of Z)
- **Repro:** `bc_dcf(3, 0.45, ...)` → derivative evaluated at η ≈ 0.4498
- **Impact:** BC c(r) inaccurate; Z is analytic, so the derivative should be too
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D036_analytic_compressibility_derivative; seen failing on 363ae8f: d=3 eta=0.45 36.879 vs 37.225; d=2 eta=0.85 546.53 vs 557.78
- **Fix:** K = d(eta Z)/d eta computed analytically from the rational Z6 equation of state

## D-037 — Lado transform sums drop the last node
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/nft/numerical_fourier_transform.py:60, :86, :108, :133, :156, :170
- **Symptom:** `range(1, N - 1)` instead of `range(1, N)`; Lado's sums run over i = 1…N−1
- **Repro:** forward then inverse transform of a Gaussian does not return the input to machine precision
- **Impact:** loss of the discrete orthogonality that makes the Lado scheme exact
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D037_lado_sums_keep_last_node; seen failing on 363ae8f: round-trip error 6.1e-2 (1D), 9.4e-1 (3D); nft(R=10, N=41) returned 500 rows
- **Fix:** theory.hankel sums all nodes (vectorised); no silent change of N

## D-038 — percus_yevick.py cannot be imported or called
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/percus_yevick.py:17-19 (bad path, `nft_1d` not exported), :125/:127 (`D=` passed to functions without `D`), :45 (array passed where callable expected); d = 2 returns None
- **Symptom:** ImportError on import; TypeError if fixed
- **Repro:** `import utils.approx.percus_yevick` → ImportError
- **Impact:** PY reference S(k) unavailable
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D038_percus_yevick_importable_and_2d_raises; seen failing on 363ae8f: ModuleNotFoundError; with imports patched py_dcf(2, 0.3) -> None and d=1 TypeError (unexpected 'D')
- **Fix:** theory.percus_yevick rewritten; d=2 raises NotImplementedError pointing to baus_colot

## D-039 — PY and BC disagree on the ρ convention of c̃(k)
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py (`bc_dcf_*` returns ρc̃), utils/approx/percus_yevick.py (`py_dcf_*` returns c̃)
- **Symptom:** S(k) = 1/(1 − ρc̃) must be formed differently for each; tmp scripts multiply ρ twice
- **Repro:** see utils/nft/tmp-2.py:510, :519
- **Impact:** comparisons between closures are wrong by a factor ρ
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D039_dcf_fourier_excludes_density; seen failing on 363ae8f: BC 1D ck / exact c~ = 0.4 (= eta)
- **Fix:** all theory functions return c~ without rho; S = 1 / (1 - rho c~)

## D-040 — Approx ABC makes its dispatcher abstract
- **Found:** 2026-10-26, during the repository audit
- **Status:** fixed
- **Where:** utils/approx/approx_class.py (`apx_dcf` abstract; returns `1` for unsupported d)
- **Symptom:** subclasses must re-implement the dispatcher; unsupported d returns 1 instead of raising
- **Repro:** n/a (no subclasses exist)
- **Impact:** design defect only
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D040_unsupported_dimension_raises; seen failing on 363ae8f: Full().apx_dcf(4, 0.5) -> 1
- **Fix:** ABC removed; module functions dispatch on d and raise ValueError for d not in {1, 2, 3}

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

## D-042 — 2D Baus–Colot c̃(k) calls np.trapz (removed in NumPy 2)
- **Found:** 2026-10-26, while porting the theory code (subagent)
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:473 (`bc_dcf`, d = 2)
- **Symptom:** `np.trapz` no longer exists → AttributeError
- **Repro:** `bc_dcf(2, 0.4, ...)` under NumPy ≥ 2.0 → AttributeError
- **Impact:** 2D BC c̃ unavailable
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D042_bc_2d_fourier_numpy2 (+ property test_bc_fourier_matches_quadrature); seen failing on 363ae8f: AttributeError: module 'numpy' has no attribute 'trapz'
- **Fix:** 2D BC c~ = analytic disk-overlap transform minus Gauss-Legendre core correction

## D-043 — bc_dcf misreads the fsolve status code
- **Found:** 2026-10-26, while porting the theory code (subagent)
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:452 (`if not ier`)
- **Symptom:** fsolve success is `ier == 1`; failure codes 2–5 are truthy, so failures pass silently and the initial guess is used; `arcsin(1/x)` evaluated for x < 1
- **Repro:** 2D, η = 0.8 → fsolve returns ier = 5, a = 1.5 used silently (true a(0.8) = 1.1146)
- **Impact:** wrong 2D BC c(r) at high packing fraction, no warning
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D043_bc_2d_a_converges_at_high_density (+ property test_bc_2d_a_is_root_up_to_high_density); seen failing on 363ae8f: eta=0.8/0.85/0.9 legacy a=1.5 (ier=5 accepted)
- **Fix:** safeguarded Newton solve with residual check; ValueError if no root

## D-044 — bc_dcf silently replaces r or q when their types differ
- **Found:** 2026-10-26, while porting the theory code (subagent)
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:416-420
- **Symptom:** if `type(r) != type(q)` the user's r or q is overwritten by a linspace
- **Repro:** `bc_dcf(3, 0.3, r=0.5, q=[1, 2, 3])` → returned c(r) is not at r = 0.5
- **Impact:** results at grid points the caller did not ask for
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D044_outputs_at_caller_grid (+ property test_bc_evaluates_at_caller_points); seen failing on 363ae8f: bc_dcf(3, 0.3, r=0.5, q=[1,2,3]) returned c(r) of shape (3,)
- **Fix:** functions take r and k separately and evaluate exactly there, preserving shape

## D-045 — closed-form c̃(k) is 0/0 at k = 0 and inaccurate at small k
- **Found:** 2026-10-26, while porting the theory code (subagent)
- **Status:** fixed
- **Where:** utils/approx/percus_yevick.py:69-82 (also `bc_dcf_1d/3d`)
- **Symptom:** catastrophic cancellation in the closed form as k → 0
- **Repro:** `py_dcf_3d(0.4, q=1e-3)` → −2.2e5, `q=1e-5` → −104.7; exact c̃(0) = −31.416
- **Impact:** S(k → 0), i.e. the compressibility, is wrong
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D045_small_k_no_cancellation (+ properties test_bc_small_k_taylor, test_py_small_k_taylor); seen failing on 363ae8f: py_dcf_3d q=0 -> nan, 1e-3 -> -223244.5, 1e-5 -> -104.7 (exact -31.416)
- **Fix:** theory.hankel.ball_polynomial_transform switches to a Taylor series for kD < 2

## D-046 — bc_dcf ignores the diameter D
- **Found:** 2026-10-26, while porting the theory code (subagent)
- **Status:** fixed
- **Where:** utils/approx/baus_colot.py:375 (`bc_dcf`)
- **Symptom:** output for D = 2 is identical to D = 1; c(1.5) = 0 although 1.5 < D
- **Repro:** `bc_dcf(3, 0.3, r=1.5, D=2.0)` → 0
- **Impact:** any non-unit diameter gives wrong c(r), c̃(k)
- **Regression test:** tests/regression/test_theory_regressions.py::test_regression_D046_diameter_is_used (+ properties test_bc_diameter_scaling, test_dcf_real_space_scales_with_diameter); seen failing on 363ae8f: c(r=1.5; D=2) = -0.0
- **Fix:** diameter scales r and k: c~(k; D) = D^d c~(kD; 1)

## D-047 — default S(k) shell width drops the smallest |k| in sheared cells
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** fixed
- **Where:** src/npointstats/structure_factor.py (`default_dk`), src/npointstats/ensemble.py (`Ensemble.structure_factor` default dk)
- **Symptom:** `default_dk` is the smallest *basis-row* norm of B, not the shortest reciprocal vector; in a non-reduced cell shorter vectors fall below the first edge 0.5·dk and are discarded. The ensemble default (max over members) can drop them for every member, contradicting its docstring.
- **Repro:** `Box([[30,0],[27,3]])`, N = 500 → `radial_average()` uses dk = 1.897 while min|k| = 0.296; 6 vectors below the first edge are lost
- **Impact:** binned S(k) misses exactly the small-k data used for hyperuniformity in sheared/Lees–Edwards cells
- **Regression test:** tests/regression/test_regression_review.py::test_regression_D047_default_dk_keeps_smallest_k; seen failing before the fix: dk 1.8966 vs shortest |k| 0.2962
- **Fix:** default dk = Box.shortest_reciprocal_norm; ensemble default = finest member spacing

## D-048 — inscribed radius and NUFFT grid depend on the cell representation
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** fixed
- **Where:** src/npointstats/box.py (`inscribed_radius`, `minimum_image`), src/npointstats/structure_factor.py (`_index_bounds`)
- **Symptom:** limits and index bounds are computed from the user's basis; a non-reduced basis of the same lattice gives a needlessly small r_max limit and a needlessly large mode grid
- **Repro:** `Box([[30,0],[27,3]]).inscribed_radius` → 1.5, while the same lattice in reduced form allows ~2.1
- **Impact:** g2 / number-variance range lost and S(k) slower for sheared cells; results otherwise correct
- **Regression test:** tests/regression/test_regression_review.py::test_regression_D048_limits_independent_of_basis (+ property test_regression_D048_change_of_basis_invariance); seen failing before the fix: inscribed radius 1.5 for the sheared basis vs 1.358 vs 1.5 across equivalent bases
- **Fix:** Box LLL-reduces (delta 0.99) plus a greedy Minkowski pass; minimum image, inscribed radius, NUFFT index grid and image shifts use the reduced basis (3000-trial stress check: 0 mismatches)

## D-049 — --raw-sk output files overwrite each other for dotted names
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/cli.py (`run`, `Path(name).stem`)
- **Symptom:** `.stem` cuts at the last dot, so `cco__phi0.45` and `cco__phi0.50` both map to `cco__phi0.npz`; exit status 0
- **Repro:** two members with those names → one file in `sk_raw/`
- **Impact:** silent loss of raw S(k) output
- **Regression test:** none yet
- **Fix:** none yet

## D-050 — single-member ensembles: wrong `counts` meaning; g2 = 0 bins dropped from CSV
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/ensemble.py (`average_binned` 1-member shortcut), src/npointstats/results.py (`save_result(nonempty_only=True)`)
- **Symptom:** with one member, `counts` holds pair counts instead of member counts and `n_members` is missing; `save_result` then removes zero-count rows, deleting the g2 = 0 (hard-core) region
- **Repro:** `npointstats -d one_lattice_file -s g2 --dr 0.25` → 12 of 20 bins written, with gaps
- **Impact:** misleading single-configuration g2 files
- **Regression test:** none yet
- **Fix:** none yet

## D-051 — documented pandas recipe loses the first data row
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** README.md (Command line), src/npointstats/results.py (module docstring)
- **Symptom:** the column names are on a `#` comment line, so `pandas.read_csv(path, comment="#")` takes the first data row as the header
- **Repro:** 12-row file → DataFrame of shape (11, 7) with columns `['0.75', '1', ...]`
- **Impact:** silently wrong data for pandas users
- **Regression test:** none yet
- **Fix:** none yet

## D-052 — submodules shadowed by same-named functions
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** fixed
- **Where:** src/npointstats/__init__.py (re-exports `structure_factor`, `pair_correlation`, `number_variance`)
- **Symptom:** `import npointstats.structure_factor as m` yields the function, so module helpers and monkeypatching are unreachable that way
- **Repro:** `import npointstats.pair_correlation as m; m.pair_counts` → AttributeError
- **Impact:** confusing API; module-level helpers inaccessible by the obvious import
- **Regression test:** tests/regression/test_regression_review.py::test_regression_D052_no_submodule_shadowing; seen failing before the fix: npointstats.number_variance is shadowed by <function number_variance>
- **Fix:** modules renamed: structure_factor.py -> structure.py, pair_correlation.py -> pairs.py, number_variance.py -> variance.py; functions still exported at top level

## D-053 — uniform_edges silently extends the range when the width exceeds it
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/binning.py (`uniform_edges`)
- **Symptom:** at least one bin is always produced, so `dr > r_max` extends r_max to dr (or raises a misleading inscribed-radius error)
- **Repro:** `pair_correlation(p, dr=0.4, r_max=0.3)` → edges `[0, 0.4]`, metadata r_max = 0.4
- **Impact:** results beyond the requested (possibly invalid) range
- **Regression test:** none yet
- **Fix:** none yet

## D-054 — CLI accepts --fractional for formats that ignore it
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/cli.py (`_reader_kwargs`)
- **Symptom:** `-p cco --fractional` exits 0 and reads Cartesian coordinates
- **Repro:** `npointstats -d dir -p cco --fractional -s g2` → exit 0
- **Impact:** the user believes coordinates were converted
- **Regression test:** none yet
- **Fix:** none yet

## D-055 — morsefire reader raises a bare KeyError for incomplete scalars.dat
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/io.py (`read_morsefire`)
- **Symptom:** missing `nDim`/`numParticles` → `KeyError: 'nDim'` traceback from the CLI
- **Repro:** `scalars.dat` without `nDim` → KeyError
- **Impact:** unhelpful error
- **Regression test:** none yet
- **Fix:** none yet

## D-056 — minor API/CLI inconsistencies
- **Found:** 2026-10-26, code review of the v0.1 refactor
- **Status:** open
- **Where:** src/npointstats/io.py (`read_simple(columns=...)` can only equal d), src/npointstats/cli.py (`--r-max` help says L/2, `--dk` help says 2π/L; `--raw-sk` uses each member's default k_max, not the k_max of `sf.csv`), src/npointstats/ensemble.py (`average_binned` compares edges with `allclose`)
- **Symptom:** parameters or help text that do not match behaviour
- **Repro:** `read_simple(path, np.eye(2), columns=1)` → ValueError although documented as "use only the first columns"
- **Impact:** confusion; raw and binned S(k) cover different k ranges
- **Regression test:** none yet
- **Fix:** none yet
