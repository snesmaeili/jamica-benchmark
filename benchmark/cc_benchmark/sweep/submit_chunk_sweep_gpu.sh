#!/bin/bash
# jamica rows of the GPU block-size sweep (Figures 5 and 6, Supplementary Figure
# S5): 25 ds004505 recordings x {1K, 4K, 16K, 64K, 262K, 512K, 1M, full batch}
# at a matched 3,000 iterations (early stopping disabled), plus the iteration
# ladder {100, 250, 500, 1000, 2000} at chunk 65,536, one H100 per task. The
# other implementations' rows are reused from the published campaign
# (benchmark/comparator/results/xperf_chunksize/raw/).
#
# Trillium-GPU: nodes carry 4 x H100 (96 cores, 745 GiB); one GPU is requested
# with --gpus-per-node=1 and a quarter of the cores. Do NOT pass --mem: Trillium
# rejects it and grants 186 GiB of host memory per GPU automatically. Submit from
# THIS directory (benchmark/cc_benchmark/sweep/) after build_sweep_venv.sh (login,
# once, SWEEP_GPU=1) and after staging ds004505 raw_bids (BIDS_ROOT_DS4505 in
# ../env.local). Override the account on the command line if rrg is justified:
#   sbatch --account=rrg-kjerbi submit_chunk_sweep_gpu.sh
#
#SBATCH --job-name=jamica_sweep_gpu
#SBATCH --account=def-kjerbi
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:30:00
#SBATCH --array=1-25%8
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/sweep/
source ../fir_env.sh || exit 1  # modules (env.local may rename cuda/cudnn) + .venv_fir + BIDS roots
REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"
source ../assert_jamica.sh || exit 1

# The fit must run on the GPU: a CPU-only jaxlib produces a complete-looking sweep
# at the wrong speed, so this is fatal.
JAX_PLATFORMS=cuda "$AMICA_PYTHON_VENV" - <<'PYCHK' || { echo "FATAL: jamica venv cannot see the GPU" >&2; exit 1; }
import jax, sys
devs = jax.devices()
print("jax", jax.__version__, "devices:", devs)
sys.exit(0 if any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in devs) else 1)
PYCHK
nvidia-smi --query-gpu=name,memory.used,memory.total,compute_mode --format=csv,noheader || true

export SWEEP_RESULTS_DIR="${SWEEP_RESULTS_DIR:-/scratch/$USER/jamica_v030/sweep}"
mkdir -p "$SWEEP_RESULTS_DIR"
echo "=== GPU block-size sweep: ds004505 sub-$SLURM_ARRAY_TASK_ID, 64 components, 3000 iterations ==="
python chunk_sweep_cell.py \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components 64 --device gpu --max-iter 3000 \
    --chunks 1024,4096,16384,65536,262144,524288,1048576,full \
    --ladder-iters 100,250,500,1000,2000 --ladder-chunk 65536 \
    --out-root "$SWEEP_RESULTS_DIR" --skip-existing
echo "=== DONE sub-$SLURM_ARRAY_TASK_ID. Cells under $SWEEP_RESULTS_DIR/gpu/ ==="
