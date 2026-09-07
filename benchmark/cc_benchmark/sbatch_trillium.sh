#!/bin/bash
# Submit one of the fir GPU scripts on Trillium-GPU without editing it.
#
# Trillium's sbatch rejects --mem (host memory is fixed at 186 GiB per GPU) and has
# no gpubase partition / h100 gres: a GPU is requested with --gpus-per-node. This
# wrapper copies the script, drops the fir-only header lines (--mem, --partition,
# --gres, --account, --cpus-per-task) and submits the copy from the CURRENT
# directory with Trillium's own options, so SLURM_SUBMIT_DIR and every relative
# `source fir_env.sh` inside the script resolve exactly as they do on fir.
#
#   cd benchmark/cc_benchmark && bash sbatch_trillium.sh submit_seed_robustness.sh --array=1-125%8
#
# Extra arguments go to sbatch verbatim (array ranges, --dependency, --export).
# Account defaults to def-kjerbi; override with TRILLIUM_ACCOUNT=rrg-kjerbi.
set -euo pipefail
script="${1:?usage: sbatch_trillium.sh <submit_script.sh> [sbatch options...]}"; shift
[ -f "$script" ] || { echo "no such script: $script" >&2; exit 1; }
mkdir -p .trillium_submit
copy=".trillium_submit/$(basename "$script")"
grep -vE '^#SBATCH --(mem|partition|gres|account|cpus-per-task)=' "$script" > "$copy"
chmod +x "$copy"
sbatch --account="${TRILLIUM_ACCOUNT:-def-kjerbi}" --gpus-per-node=1 --cpus-per-task="${TRILLIUM_CPUS:-24}" "$@" "$copy"
