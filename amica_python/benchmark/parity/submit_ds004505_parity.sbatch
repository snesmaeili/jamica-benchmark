#!/bin/bash
#SBATCH --job-name=amica17_ds4505_parity
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

# Submit FROM amica_python/benchmark/parity/ (three levels below the repo root).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$SUBMIT_DIR/../../.." && pwd)"
BIN="${AMICA17_BIN:-/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17}"     # archived, validated
WORK="${AMICA_PARITY_WORK:-/scratch/$USER/jamica_v030/fortran_parity/ds004505_sub01}"
ENVSH="$REPO/benchmark/cc_benchmark/fir_env.sh"

echo "############ PHASE 0: prep ds004505 sub-01 → PCA-64 (amica-python env) ############"
module purge
source "$ENVSH"
# amica_python is the vendored harness at the repo root, not a pip-installed
# package in .venv_fir, so put the repo root on the path for the `python -m` calls.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python -c "import sys, mne, numpy, jamica; print('py', sys.executable, 'mne', mne.__version__, '| jamica', jamica.__version__, jamica.__file__)"
mkdir -p "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity prep \
    --workdir "$WORK" --dataset ds004505 --subject 1 --k-pca 64 \
    --m 3 --do-newton 1 --max-iter 100 \
    --container-data "$WORK/data" --container-out "$WORK/out"

echo "############ PHASE 1: run archived amica17 (gcc/openmpi env) ############"
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas
export OMP_NUM_THREADS=1
file "$BIN"
cd "$WORK"
rm -rf "$WORK/out"
mpirun -np 1 "$BIN" "$WORK/amica.param"
echo "----- Fortran out.txt (last 3 lines) -----"
tail -3 "$WORK/out/out.txt" || true

echo "############ PHASE 2: amica-python compare (amica-python env) ############"
module purge
source "$ENVSH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$WORK"
python -m amica_python.benchmark.parity.run_fortran_parity compare --workdir "$WORK"
cp -f "$WORK/parity.json" "$WORK/parity_ds004505_sub01.json" 2>/dev/null || true
echo "############ DONE ############"
