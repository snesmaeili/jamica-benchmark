#!/bin/bash
# Throttled job-array version of the atomic CPU cells, for CLEANER absolute timing
# WITHOUT hogging nodes (fair-share-neutral): a %N concurrency cap keeps only a few
# 8-core bycore cells running at once, so our own cells rarely co-locate and starve
# each other of a node's shared DRAM bandwidth. Same footprint as the un-throttled
# run, just spread over time. Reuses the per-subject caches (no re-preprocess).
#
# Launch (from cc_benchmark/ on fir), e.g. 125 cells, 6 at a time:
#   MANIFEST=/scratch/yorguin/realchunk_cells/manifest.txt \
#     sbatch --array=1-125%6 submit_cell_array.sh
# manifest.txt: one "SUBJECT IMPL CHUNK" per line (built by the launcher).
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yorguin/realchunk_cells/arr-%A_%a.out
#SBATCH --error=/scratch/yorguin/realchunk_cells/arr-%A_%a.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR"
source fir_env.sh || exit 1
export AMICA_SKIP_PIN_CHECK=1
: "${MANIFEST:?set MANIFEST}"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$line" ] && { echo "no manifest line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
read -r S IMPL C <<< "$line"
st=$(printf "sub-%02d" "$S")
R=/scratch/yorguin/amica-benchmark-repro
CACHE=/scratch/yorguin/realchunk_cache/ds004505_${st}_nc64.npz
ALL="amica_python_jax amica_python_jax_chunked amica_python_numpy neuromechanist_numpy pyamica_torch scott_huberty_torch pamica_torch fortran_amica17"
SKIP=""; for k in $ALL; do [ "$k" = "$IMPL" ] || SKIP="$SKIP $k"; done
export AMICA_PYTHON_VENV=$R/.venv_fir_gpu/bin/python
export COMPETITORS_VENV=$R/.venv_competitors_main/bin/python
export PAMICA_VENV=$R/.venv_pamica_main/bin/python
export AMICA_SRC=/scratch/yorguin/amica_main_src
export AMICA_SCOTT_BATCH=$C AMICA_PYAMICA_CHUNK=$C AMICA_PAMICA_BLOCK_SIZE=$C AMICA_FORTRAN_BLOCK=$C
export AMICA_COMPARATOR_RESULTS=/scratch/yorguin/xperf_realchunk_main/cpu_throttled
mkdir -p "$AMICA_COMPARATOR_RESULTS"
FOPT=""
if [ "$IMPL" = fortran_amica17 ]; then
  FOPT="--include-fortran"
  export AMICA17_BIN=/project/rrg-kjerbi/yorguin/amica_fortran_reference/amica17
fi
echo "=== array cell: sub-$S impl=$IMPL chunk=$C (skip complement) ==="
python ../comparator/implementation_perf.py \
    --dataset ds004505 --subject "$S" --input-level bids \
    --n-components 64 --max-iter 100 --seeds 0 \
    --amica-device cpu --competitor-device cpu \
    --amica-chunk-size "$C" \
    --cached-input "$CACHE" \
    --out-tag "c$C" $FOPT \
    --skip $SKIP
echo "ARRAY_CELL_DONE sub-$S $IMPL c$C"
