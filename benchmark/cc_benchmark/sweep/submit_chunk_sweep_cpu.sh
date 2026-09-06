#!/bin/bash
# jamica rows of the CPU block-size sweep (Figures 5 and 6, Supplementary Figure
# S5): 25 ds004505 recordings x {1K, 4K, 16K, 64K, 262K, 512K, 1M, full batch}
# at a matched 250 iterations (early stopping disabled), plus the iteration
# ladder {50, 100, 500} at chunk 65,536. One fit per WHOLE node (exclusive), so
# no memory-bandwidth contention from neighbours, as in the published Narval
# campaign whose other implementations' rows are reused.
#
# Narval CPU nodes: 64 cores (Zen2/3/4), 249+ GB. Submit from THIS directory
# (benchmark/cc_benchmark/sweep/) after build_sweep_venv.sh (login, once) and
# after staging ds004505 raw_bids (BIDS_ROOT_DS4505 in ../env.local).
#
#SBATCH --job-name=jamica_sweep_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --array=1-25%6
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/sweep/
source ../fir_env.sh || exit 1  # modules + .venv_fir + BIDS roots (env.local); OMP threads = 64
REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"
source ../assert_jamica.sh || exit 1
echo "node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  mem: $(free -g | awk '/Mem:/{print $2}') GB"

export SWEEP_RESULTS_DIR="${SWEEP_RESULTS_DIR:-/scratch/$USER/jamica_v030/sweep}"
mkdir -p "$SWEEP_RESULTS_DIR"
echo "=== CPU block-size sweep: ds004505 sub-$SLURM_ARRAY_TASK_ID, 64 components, 250 iterations, whole node ==="
python chunk_sweep_cell.py \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components 64 --device cpu --max-iter 250 \
    --chunks 1024,4096,16384,65536,262144,524288,1048576,full \
    --ladder-iters 50,100,500 --ladder-chunk 65536 \
    --out-root "$SWEEP_RESULTS_DIR" --skip-existing
echo "=== DONE sub-$SLURM_ARRAY_TASK_ID. Cells under $SWEEP_RESULTS_DIR/cpu/ ==="
