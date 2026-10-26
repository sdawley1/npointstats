#!/bin/bash
# Old-vs-new benchmark on one exclusive-ish compute node (CPU only).
#SBATCH --job-name=npst-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=benchmarks/bench-%j.log

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1   # same thread budget for every implementation
NEW=/home/sd2402/micromamba/envs/npstats/bin/python
OLD=/home/sd2402/micromamba/envs/npstats-legacy/bin/python
lscpu | grep -E "Model name|^CPU\(s\)"
"$NEW" benchmarks/bench_old_vs_new.py --legacy-python "$OLD" --new-python "$NEW" \
    --out "benchmarks/results-${SLURM_JOB_ID}.json"
