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
source fir_env.sh || exit 1               # modules (incl. cuda/cudnn) + .venv_fir + env.local

# --- which jamica is measured -------------------------------------------------
# Default: the `jamica` release installed in this checkout's .venv_fir by
# fir_env.sh (pinned in pins.toml, [[venv]] fir). Set AMICA_SRC to a source
# checkout to measure that instead: implementation_perf.py puts it on PYTHONPATH
# for OUR runner only. It must not go on PYTHONPATH globally -- scott-huberty's
# package is a different project (imported as `amica`), and a global path would
# shadow whatever shares a module name with the checkout.
REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"

# The competitor venvs were built by setup_competitors.sh / setup_pamica.sh under
# the older clones on fir; override per site. Left unset, every competitor run
# dies instantly with "venv python missing" and the task still exits 0, so the
# array looks like it succeeded while producing nothing.
export COMPETITORS_VENV="${COMPETITORS_VENV:-/scratch/$USER/jamica/.venv_competitors/bin/python}"
export PAMICA_VENV="${PAMICA_VENV:-/scratch/$USER/jamica-benchmark/.venv_pamica/bin/python}"
for _v in "$AMICA_PYTHON_VENV" "$COMPETITORS_VENV" "$PAMICA_VENV"; do
    [ -x "$_v" ] || { echo "FATAL: no interpreter at $_v" >&2; exit 1; }
done

# installed == intended: assert each competitor venv holds the pinned commit
# from pins.toml before the first fit. Catches silent upstream HEAD drift and a
# clobbered install (e.g. the pyamica/pyAMICA name collision).
"$COMPETITORS_VENV" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv competitors || exit 1
"$PAMICA_VENV"      "$SLURM_SUBMIT_DIR/check_env.py" verify --venv pamica      || exit 1

# The measured jamica: the pinned release imported from the venv, or AMICA_SRC at
# the pinned commit (assert_jamica.sh). Fails fast rather than benchmark the
# wrong code, and prints the identity into the job log.
source "$SLURM_SUBMIT_DIR/assert_jamica.sh" || exit 1
# ---------------------------------------------------------------------------

ITERS=(100 400 700 1000)
IT="${ITERS[$SLURM_ARRAY_TASK_ID]}"

# Walltime justification. The only archived GPU per-iteration cost that is not
# self-contradictory is pamica_torch at 1.88 s/iter, which sets the ceiling:
# ~31 min at 1000 iterations. The other three are well under a minute per run at
# any of these caps. 3 h is deliberate headroom, because the archived GPU
# timings are exactly what this job exists to distrust.
export AMICA_COMPARATOR_RESULTS="${AMICA_MEM_RESULTS:-${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}}/itercurve/gpu"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== GPU iteration curve: max_iter=$IT ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,compute_mode --format=csv,noheader || true

# Check the interpreters that actually run the fits, not the one fir_env.sh
# happens to activate. That venv carries a CPU-only jaxlib, so probing it prints
# "[CpuDevice(id=0)]" on a perfectly good GPU node -- which is exactly what made
# a working allocation look broken once already. A GPU panel built from runners
# that silently fell back to CPU would be worse than no panel, so this is fatal.
gpu_check() {
    "$1" - "$2" <<'PYCHK'
import importlib.util as u, sys
label = sys.argv[1]
ok = True
if u.find_spec("jax") is not None:
    import jax
    devs = jax.devices()
    ok = any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in devs)
    print(f"  {label} jax devices: {devs}")
if u.find_spec("torch") is not None:
    import torch
    ok = torch.cuda.is_available()
    print(f"  {label} torch {torch.__version__} cuda: {ok}")
    if ok:
        torch.zeros(1024, 1024, device="cuda"); torch.cuda.synchronize()
sys.exit(0 if ok else 1)
PYCHK
}
gpu_check "$AMICA_PYTHON_VENV" jamica       || { echo "FATAL: jamica venv cannot see the GPU" >&2; exit 1; }
gpu_check "$COMPETITORS_VENV"  competitors || { echo "FATAL: competitors venv cannot see the GPU" >&2; exit 1; }
gpu_check "$PAMICA_VENV"       pamica      || { echo "FATAL: pamica venv cannot see the GPU" >&2; exit 1; }

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
