#!/bin/bash
# Synthetic multi-model controls (Supplementary Table S7, Figure 7 synthetic
# panels): the three-regime non-stationary fixture and its stationary control,
# H = 1..10 at 2,000 iterations, ten seeds (array 0-9), on the released jamica.
# Seed s writes <MM_RESULTS_DIR>/synthetic/seed<s>/synthetic_summary.json, the
# layout make_tab_multimodel_seeds.py reads (seed0 = the main-text control).
#
# Submit from this directory on the fir login node:
#   sbatch submit_multimodel_synthetic.sh
#
#SBATCH --job-name=mm_synthetic
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"          # scripts/multimodel/paper_protocol/
REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/../../.." && pwd)"
source "$REPO_ROOT/benchmark/cc_benchmark/fir_env.sh"

S="$SLURM_ARRAY_TASK_ID"
OUT="${MM_RESULTS_DIR:-/scratch/$USER/jamica_v030/multimodel}/synthetic/seed${S}"
mkdir -p "$OUT"
echo "seed $S -> $OUT"
JAX_PLATFORMS=cuda python run_synthetic_multimodel.py \
    --n-components 16 --tseg 25000 --n-regimes 3 --max-h 10 --max-iter 2000 \
    --seed "$S" --sfreq 250 --out-dir "$OUT"
echo "=== DONE seed $S ==="
