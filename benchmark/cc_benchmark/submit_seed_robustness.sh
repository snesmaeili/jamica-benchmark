#!/bin/bash
# Initialisation robustness of the single-model benchmark (Discussion, Table
# tennis): the 25 ds004505 participants refitted from five random seeds each,
# same protocol as submit_jax_gpu_v3.sh (3,000 iterations, 64 PCs, JAX-GPU).
# The v3 JSON per (subject, seed) carries MIR + the provenance block; the
# within-participant SD of MIR across seeds is computed off-cluster.
#
# Array mapping (125 tasks): task t -> subject = (t-1)/5 + 1, seed = (t-1) % 5.
# Seeds 0-4 land in separate directories because the runner names its output
# from dataset/subject/backend/device only.
#
# ds004505 is stored at 250 Hz, so --resample 250 is a no-op kept for protocol
# fidelity with the archived seed-robustness campaign.
#
#SBATCH --job-name=jamica_seed_robustness
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-125%25
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh               # modules + .venv_fir (jamica pinned) + env.local

T="$SLURM_ARRAY_TASK_ID"
SID=$(( (T - 1) / 5 + 1 ))
SEED=$(( (T - 1) % 5 ))
export AMICA_RESULTS_DIR="${SEED_RESULTS_DIR:-/scratch/$USER/jamica_v030/seed_robustness}/seed${SEED}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
mkdir -p "$AMICA_RESULTS_DIR"
echo "task $T -> subject $SID, seed $SEED -> $AMICA_RESULTS_DIR"

python run_one_subject.py \
    --dataset ds004505 --subject "$SID" \
    --device gpu --backend jax \
    --n-iter "${AMICA_N_ITER:-3000}" --n-components 64 --resample 250 \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --random-state "$SEED" \
    --schema-version v3 \
    --output-dir "$AMICA_RESULTS_DIR"
echo "=== DONE sub-${SID} seed ${SEED} ==="
