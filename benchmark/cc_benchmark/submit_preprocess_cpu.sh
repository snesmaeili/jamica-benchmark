#!/bin/bash
# One-time per-subject preprocessing for the atomic chunk-size cells: run the PCA
# projection ONCE and cache it to AMICA_CACHED_INPUT, so the fan-out of atomic
# (impl, chunk) cells reuses the same projected array instead of re-preprocessing.
# Small + fast (no fit). Submit one per subject; cells depend on it via afterok.
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yorguin/realchunk_cells/%x-%j.out
#SBATCH --error=/scratch/yorguin/realchunk_cells/%x-%j.err
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source fir_env.sh || exit 1           # modules + BIDS_ROOT + venvs
export AMICA_SKIP_PIN_CHECK=1         # amica fit runs in AMICA_PYTHON_VENV/AMICA_SRC, not this venv
: "${AMICA_CACHED_INPUT:?set AMICA_CACHED_INPUT}"; : "${AMICA_MEM_SUBJECT:?set AMICA_MEM_SUBJECT}"
mkdir -p "$(dirname "$AMICA_CACHED_INPUT")"
echo "=== preprocess-only: ${AMICA_MEM_DATASET:-ds004505} sub-${AMICA_MEM_SUBJECT} -> $AMICA_CACHED_INPUT ==="
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --resample-sfreq "${AMICA_RESAMPLE_SFREQ:-250}" \
    --seeds 0 \
    --cached-input "$AMICA_CACHED_INPUT" \
    --preprocess-only
echo "PREPROCESS_DONE"
