#!/bin/bash
# Multi-model H-sweep (Figure 7, Supplementary Tables S6-S7, Figure S4) on the
# released jamica: one cohort per invocation, a 1-task smoke (first subject, H=2)
# gating the array via afterok. Drives run_multimodel_benchmark.py on fir GPUs
# with the protocol of the manuscript: 16 PCs, full recording, 250 Hz, 2,000
# iterations, 3 mixture components, seed 0, per-site mains notch.
#
# Env vars:
#   DS          dataset (ds004505 | ds004504 | ds004621)
#   SUB_BASE    first subject id (1 for ds004505/ds004621, 37 for ds004504)
#   NSUB        number of subjects
#   OUTSUB      output subdir under $MM_RESULTS_DIR (default /scratch/$USER/jamica_v030/multimodel)
#   JOBNAME     slurm job name
#   HMAX        max H (default 10);  EXTRA  extra runner args;  INPUT_LEVEL (bids);
#   GATE        1 = smoke-gate the array (default 1);  SEED (0)
# Array = NSUB*HMAX; task t -> subject = SUB_BASE + (t-1)/HMAX, H = (t-1)%HMAX + 1.
#
# The six cohorts of the manuscript (output subdirs match the producers' figdata names):
#   DS=ds004505 SUB_BASE=1  NSUB=25 OUTSUB=mmbench_ds004505      JOBNAME=mm_ds004505      bash submit_multimodel_cohort.sh
#   DS=ds004505 SUB_BASE=1  NSUB=25 OUTSUB=mmbench_ds004505_ch19 JOBNAME=mm_ds004505_ch19 EXTRA="--channel-subset tentwenty" bash submit_multimodel_cohort.sh
#   DS=ds004504 SUB_BASE=37 NSUB=29 OUTSUB=mmbench_ds004504      JOBNAME=mm_ds004504      bash submit_multimodel_cohort.sh
#   DS=ds004621 SUB_BASE=1  NSUB=25 OUTSUB=mmbench_ds004621      JOBNAME=mm_ds004621      bash submit_multimodel_cohort.sh
#   DS=ds004505 SUB_BASE=1  NSUB=5  OUTSUB=mmbench_ds004505_surr JOBNAME=mm_ds004505_surr EXTRA="--surrogate phase" bash submit_multimodel_cohort.sh
#   DS=ds004504 SUB_BASE=37 NSUB=5  OUTSUB=mmbench_ds004504_surr JOBNAME=mm_ds004504_surr EXTRA="--surrogate phase" bash submit_multimodel_cohort.sh
#
# Run ON the fir login node (login-safe; only sbatch) from this directory.
set -euo pipefail
: "${DS:?set DS}"; : "${SUB_BASE:?set SUB_BASE}"; : "${NSUB:?set NSUB}"
: "${OUTSUB:?set OUTSUB}"; : "${JOBNAME:?set JOBNAME}"
HMAX=${HMAX:-10}; EXTRA=${EXTRA:-}; INPUT_LEVEL=${INPUT_LEVEL:-bids}; GATE=${GATE:-1}; SEED=${SEED:-0}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # scripts/multimodel/paper_protocol
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
CC="$REPO_ROOT/benchmark/cc_benchmark"
[ -f "$CC/env.local" ] && source "$CC/env.local"
GPU_ACCOUNT="${AMICA_GPU_ACCOUNT:-def-kjerbi_gpu}"
GPU_PARTITION="${AMICA_GPU_PARTITION:-gpubase_bygpu_b1}"
GPU_GRES="${AMICA_GPU_GRES:-gpu:h100:1}"
OUT="${MM_RESULTS_DIR:-/scratch/$USER/jamica_v030/multimodel}/$OUTSUB"
mkdir -p "$OUT" "$HERE/logs"
cd "$HERE"
COMMON="--n-components 16 --duration-sec 0 --resample 250 --max-iter 2000 --num-mix 3 --seed $SEED --skip-underpowered --input-level $INPUT_LEVEL --output-dir $OUT $EXTRA"
NTASK=$((NSUB * HMAX))

DEP=""
if [ "$GATE" = "1" ]; then
  SMOKE=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}_smoke
#SBATCH --account=$GPU_ACCOUNT
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=$GPU_GRES
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd $HERE
source $CC/fir_env.sh
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset $DS --subject $SUB_BASE --num-models 2 $COMMON
EOF
)
  echo "smoke job: $SMOKE"
  DEP="--dependency=afterok:$SMOKE"
fi

sbatch $DEP <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --account=$GPU_ACCOUNT
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=$GPU_GRES
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=1-${NTASK}%30
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd $HERE
source $CC/fir_env.sh
T=\$SLURM_ARRAY_TASK_ID
SID=\$(( $SUB_BASE + (T-1)/$HMAX ))
H=\$(( (T-1)%$HMAX + 1 ))
echo "task \$T -> subject \$SID, H \$H"
JAX_PLATFORMS=cuda python run_multimodel_benchmark.py --dataset $DS --subject \$SID --num-models \$H $COMMON
EOF
echo "submitted $JOBNAME : $NTASK tasks (-> $OUT)"
