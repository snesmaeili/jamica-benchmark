#!/bin/bash
# Cross-implementation agreement on the two replication recordings (Supplementary
# Table S5), CPU half: every implementation fits the same PCA-projected recording
# for 100 iterations, and the matched-row correlations against the Fortran
# reference and against each other are computed off-cluster from the result JSONs.
#
#   task 0: ds004504 sub-37, 15 components (19-channel eyes-closed rest)
#   task 1: ds004621 sub-01, 64 components (127-channel eyes-open rest)
#
# The ds004505 sub-01 row of the same table comes from the 100-iteration task of
# submit_iter_curve_cpu.sh / submit_iter_curve_gpu.sh, so it is not repeated here.
# Result layout: $AMICA_COMPARATOR_RESULTS/<dataset>/cpu/<impl>_sub-NN_seed0_result.json
#
#SBATCH --job-name=jamica_crossrec_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1

REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"
export COMPETITORS_VENV="${COMPETITORS_VENV:-/scratch/$USER/jamica/.venv_competitors/bin/python}"
export PAMICA_VENV="${PAMICA_VENV:-/scratch/$USER/jamica-benchmark/.venv_pamica/bin/python}"
for _v in "$AMICA_PYTHON_VENV" "$COMPETITORS_VENV" "$PAMICA_VENV"; do
    [ -x "$_v" ] || { echo "FATAL: no interpreter at $_v" >&2; exit 1; }
done
"$COMPETITORS_VENV" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv competitors || exit 1
"$PAMICA_VENV"      "$SLURM_SUBMIT_DIR/check_env.py" verify --venv pamica      || exit 1
source "$SLURM_SUBMIT_DIR/assert_jamica.sh" || exit 1

case "$SLURM_ARRAY_TASK_ID" in
    0) DS=ds004504; SUBJ=37; NCOMP=15 ;;
    1) DS=ds004621; SUBJ=1;  NCOMP=64 ;;
    *) echo "FATAL: unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

export AMICA_COMPARATOR_RESULTS="${CROSSREC_RESULTS_DIR:-/scratch/$USER/jamica_v030/cross_recording}/$DS"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== cross-recording agreement (CPU): $DS sub-$SUBJ, $NCOMP components, 100 iterations ==="
python ../comparator/implementation_perf.py \
    --dataset "$DS" --subject "$SUBJ" --input-level bids \
    --n-components "$NCOMP" --max-iter 100 \
    --amica-device cpu --competitor-device cpu \
    --amica-chunk-size auto \
    --out-tag cpu \
    --skip amica_python_numpy amica_python_jax

echo "=== DONE $DS. Results under $AMICA_COMPARATOR_RESULTS/cpu/ ==="
