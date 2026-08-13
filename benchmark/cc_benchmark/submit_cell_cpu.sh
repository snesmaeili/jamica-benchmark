#!/bin/bash
# One atomic (impl, chunk) benchmark cell on CPU: load the cached projected input
# (no re-preprocessing) and fit exactly ONE implementation at one chunk size.
# Isolation (a hang/OOM wastes only this cell) + parallelism (the scheduler runs
# them independently, so the slow scott@1024 cell doesn't block amica@1024).
#
# Required env: AMICA_MEM_SUBJECT, AMICA_CACHED_INPUT, AMICA_MEM_CHUNK, CELL_SKIP
#   (space-separated impls to skip = the complement of the one kept), AMICA_COMPARATOR_RESULTS.
# The kept impl reads its own chunk knob (AMICA_MEM_CHUNK -> --amica-chunk-size for amica;
#   AMICA_SCOTT_BATCH / AMICA_PYAMICA_CHUNK / AMICA_PAMICA_BLOCK_SIZE / AMICA_FORTRAN_BLOCK for the
#   others); setting them all to the same value is harmless since only one impl runs.
# For Fortran cells set CELL_INCLUDE_FORTRAN=1 and AMICA17_BIN to the durable binary.
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yorguin/realchunk_cells/%x-%j.out
#SBATCH --error=/scratch/yorguin/realchunk_cells/%x-%j.err
set -o pipefail
cd "$SLURM_SUBMIT_DIR"                 # benchmark/cc_benchmark/
source fir_env.sh || exit 1
export AMICA_SKIP_PIN_CHECK=1
: "${AMICA_MEM_SUBJECT:?}"; : "${AMICA_CACHED_INPUT:?}"; : "${CELL_SKIP:?}"
: "${AMICA_COMPARATOR_RESULTS:?}"; mkdir -p "$AMICA_COMPARATOR_RESULTS"
FORTRAN_OPT=""; [ "${CELL_INCLUDE_FORTRAN:-0}" = "1" ] && FORTRAN_OPT="--include-fortran"
TAG_OPT=""; [ -n "${AMICA_MEM_TAG:-}" ] && TAG_OPT="--out-tag ${AMICA_MEM_TAG}"
echo "=== cell: sub-${AMICA_MEM_SUBJECT} chunk=${AMICA_MEM_CHUNK} keep=(all but: ${CELL_SKIP}) ==="
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter "${AMICA_MEM_ITER:-100}" \
    --seeds 0 \
    --amica-device cpu --competitor-device cpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    --cached-input "$AMICA_CACHED_INPUT" \
    $FORTRAN_OPT $TAG_OPT \
    --skip ${CELL_SKIP}
echo "CELL_DONE"
