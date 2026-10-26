# Defect log

## D-001 — sample_xd_sf rescales the caller's coordinates in place
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:214-243 (`sample_xd_sf`)
- **Symptom:** `np.asarray` does not copy a float64 array, so the in-place `/=` maps the caller's positions to [-pi, pi)
- **Repro:** `x = np.random.rand(10, 3); sample_xd_sf(x, np.eye(3)); x.min()` → about `-pi`, expected unchanged in [0, 1)
- **Impact:** any later use of `coords` after the call (e.g. a second S(k) or g2 on the same array) sees rescaled positions
- **Regression test:** none yet
- **Fix:** none yet

## D-002 — angular_average drops the last populated bin
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:120-130 (`angular_average`)
- **Symptom:** the loop runs over `unique.size - 1` bins and the return slices `W[:-1]`, so the bin with the largest |k| is never reported; the slicing `X[ux1:ux2]` also assumes `X` is already sorted by bin
- **Repro:** `angular_average(np.array([[0., 1.], [1., 2.], [2., 3.]]), xidx=0, yidx=1, dk=1.0)` → 1 row `[0, 1]`; the bin holding x = 1, 2 (mean S = 2.5) is lost
- **Impact:** the highest-|k| shell is missing from every binned S(k); unsorted input gives wrong means
- **Regression test:** none yet
- **Fix:** none yet

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
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:219-225 (`sample_xd_sf`)
- **Symptom:** the `match dimension` block overwrites `kn` unconditionally, so a caller-supplied grid has no effect
- **Repro:** `sample_xd_sf(x, np.eye(3), kn=(5, 5, 5)).shape[0]` → `15624` (25^3 - 1), expected `124`
- **Impact:** every 3D call uses a 25^3 grid regardless of the request
- **Regression test:** none yet
- **Fix:** none yet

## D-005 — Python module has a bash shebang
- **Found:** 2026-09-25, while porting the finufft S(k) to neurons/analyze.py
- **Status:** open
- **Where:** src/cpu_analysis/c_structure_factor.py:1
- **Symptom:** `#!/bin/bash` on a Python file; executing it directly runs bash on Python source
- **Repro:** `chmod +x c_structure_factor.py && ./c_structure_factor.py` → bash syntax errors
- **Impact:** only direct execution; `python3 c_structure_factor.py` is unaffected
- **Regression test:** none yet
- **Fix:** none yet
