#!/bin/bash
# Multi-model demo fit (Figure 7 posterior panel, Figure S4) on the released jamica:
# ds004505 sub-04, H in {1,2,3}, 64 PCs, first 600 s at 250 Hz, 2,000 iterations, three
# mixture components, seed 0 -- the archived amica-mm submit_multimodel_demo.sh protocol.
# One GPU task per H. Site-specific Slurm options come from env.local as
# AMICA_GPU_SBATCH_OPTS (fir values used when unset), like submit_multimodel_cohort.sh.
#
# Env vars:  SUBJECTS  Slurm array of subject ids (default "4");  HS  list of H (default "1 2 3")
# Run ON the login node (login-safe; only sbatch) from this directory:
#   bash submit_multimodel_demo.sh
set -euo pipefail
SUBJECTS=${SUBJECTS:-4}; HS=${HS:-"1 2 3"}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # scripts/multimodel/paper_protocol
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
CC="$REPO_ROOT/benchmark/cc_benchmark"
[ -f "$CC/env.local" ] && source "$CC/env.local"
GPU_OPTS="${AMICA_GPU_SBATCH_OPTS:---account=def-kjerbi_gpu --partition=gpubase_bygpu_b1 --gres=gpu:h100:1 --mem=24G --cpus-per-task=4}"
OUT="${MM_RESULTS_DIR:-/scratch/$USER/jamica_v030/multimodel}/multimodel_demo"
mkdir -p "$OUT" "$HERE/logs"
cd "$HERE"

for H in $HS; do
  # shellcheck disable=SC2086
  JOB=$(sbatch --parsable $GPU_OPTS --job-name="mm_demo_M${H}" --time=00:45:00 --array="$SUBJECTS" \
               --output=logs/%x-%A_%a.out --error=logs/%x-%A_%a.err <<EOF
#!/bin/bash
set -euo pipefail
cd $HERE
source $CC/fir_env.sh
JAX_PLATFORMS=cuda python run_multimodel_demo.py --subject \$SLURM_ARRAY_TASK_ID --num-models $H \
    --n-iter 2000 --n-components 64 --num-mix 3 --duration-sec 600 --resample 250 --seed 0 --output-dir $OUT
EOF
)
  echo "mm_demo_M${H}: job $JOB (subjects $SUBJECTS -> $OUT)"
done
