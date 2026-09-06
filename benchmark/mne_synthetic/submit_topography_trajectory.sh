#!/bin/bash
#SBATCH --job-name=amica_yor19_topo_traj
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# YOR19: score one continuing AMICA trajectory on the Figure 2 known-topography
# fixture. Two passes over the same fixture -- a 10,000-iteration run whose
# checkpoints land exactly on the published 3,000 and 10,000 budgets, and a
# fine-grained pass over the first 500 iterations where the likelihood flattens.
#
# Both fits reuse the archived whitener and planted maps, so the numbers are
# directly comparable to the published medians. The fixture cache already exists
# under the Figure 2 results dir, so nothing is regenerated and the MNE sample
# dataset is never downloaded.
#
# Roots follow the campaign layout: FIG2 is where submit_figure2_topography.sh
# wrote (AMICA_FIGURE2_RESULTS_DIR), OUT is this job's own directory. Submit with
# --dependency=afterok:<figure2 job>.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env_synthetic.sh

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

FIG2="${AMICA_FIGURE2_RESULTS_DIR:-/scratch/$USER/amica_figure2_topography}"
OUT="${AMICA_TRAJECTORY_RESULTS_DIR:-/scratch/$USER/jamica_v030/topography_trajectory}"
mkdir -p "$OUT"
[ -f "$FIG2/figure2_topography_fit_outputs.npz" ] || { echo "FATAL: no Figure 2 archive at $FIG2" >&2; exit 1; }

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} fig2=$FIG2 out=$OUT"
python -c "import jax, jamica; print('[job] jax devices:', jax.devices(), '| jamica', jamica.__version__, jamica.__file__)"

# A failing acceptance gate is a reportable result, not a reason to abandon the
# second pass, so neither exit code aborts the job. Both are surfaced at the end.
rc250=0
python -u run_topography_trajectory.py \
    --max-iter 10000 --writestep 250 --keep-snapshots \
    --archive "$FIG2/figure2_topography_fit_outputs.npz" \
    --cache-dir "$FIG2/cache" \
    --out-dir "$OUT/w250" || rc250=$?

rc10=0
python -u run_topography_trajectory.py \
    --max-iter 500 --writestep 10 --keep-snapshots \
    --archive "$FIG2/figure2_topography_fit_outputs.npz" \
    --cache-dir "$FIG2/cache" \
    --out-dir "$OUT/w10" || rc10=$?

echo "[job] gate exit codes: w250=$rc250 w10=$rc10 (non-zero = a gate failed)"
ls -l "$OUT/w250"/*.json "$OUT/w10"/*.json
