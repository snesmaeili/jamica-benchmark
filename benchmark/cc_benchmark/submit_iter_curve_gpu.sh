#!/bin/bash
# Cluster GPU: runtime as a function of iteration count, every GPU-capable
# implementation.
#
# Produces the GPU panel of the runtime-vs-iterations figure, and replaces the
# archived GPU runtime points, which cannot support a curve. On the archived
# runs amica_python_jax_chunked takes 39.9 s at 100 iterations and 14.4 s at 600
# -- a negative per-iteration cost -- while scott_huberty and pyamica both land
# on exactly 48.6 s at 600. At those magnitudes the measurement is dominated by
# compilation and warm-up rather than by the fit, so the archived numbers cannot
# be extended to 1000 iterations; they have to be replaced.
#
# Two things are done differently here for that reason:
#   * four iteration caps, so the per-iteration cost comes from a slope over a
#     range rather than from a difference between two warm-up-dominated points;
#   * a discarded warm-up fit before each timed one, so JIT compilation and
#     cuBLAS/cuDNN autotuning are not charged to the first timed iteration.
#
# One array task per iteration cap, all implementations inside it, so each task
# is short and a failure costs one point rather than the whole panel.
#
#SBATCH --job-name=amica_iter_curve_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh               # modules (incl. cuda/cudnn) + .venv_fir + env.local

ITERS=(100 400 700 1000)
IT="${ITERS[$SLURM_ARRAY_TASK_ID]}"

# Walltime justification. The only archived GPU per-iteration cost that is not
# self-contradictory is pamica_torch at 1.88 s/iter, which sets the ceiling:
# ~31 min at 1000 iterations. The other three are well under a minute per run at
# any of these caps. 3 h is deliberate headroom, because the archived GPU
# timings are exactly what this job exists to distrust.
export AMICA_COMPARATOR_RESULTS="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/itercurve/gpu"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== GPU iteration curve: max_iter=$IT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import jax; print('jax devices:', jax.devices())"

# Warm-up: one short fit whose timing is thrown away, so compilation and
# autotuning do not land inside the measured run.
echo "--- warm-up (discarded) ---"
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter 10 \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    --nvml-crosscheck \
    --out-tag "itercurve_gpu/warmup_iter${IT}" \
    --skip amica_python_numpy amica_python_jax > /dev/null 2>&1 || true

echo "--- timed run: max_iter=$IT ---"
python ../comparator/implementation_perf.py \
    --dataset "${AMICA_MEM_DATASET:-ds004505}" \
    --subject "${AMICA_MEM_SUBJECT:-1}" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --n-components "${AMICA_MEM_NCOMP:-64}" \
    --max-iter "$IT" \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
    --nvml-crosscheck \
    --out-tag "itercurve_gpu/iter${IT}" \
    --skip amica_python_numpy amica_python_jax

echo "=== DONE (max_iter=$IT). Results under $AMICA_COMPARATOR_RESULTS/itercurve_gpu/ ==="
