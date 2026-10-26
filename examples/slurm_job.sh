#!/bin/bash
# Example SLURM job for Della: ensemble statistics of one directory.
# Install npointstats into an environment from a login node first (compute
# nodes are offline), then submit with: sbatch examples/slurm_job.sh

#SBATCH --job-name=npointstats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate npstats
export PYTHONNOUSERSITE=1
# one BLAS/OpenMP thread per process: parallelism comes from --jobs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# one process per CPU across ensemble members; one FINUFFT thread each
npointstats -d ./patterns -s sf g2 nv \
    --jobs "${SLURM_CPUS_PER_TASK}" --nthreads 1 \
    --seed 1 --out "results-${SLURM_JOB_ID}"
