#!/bin/bash
# Cross-implementation agreement on the two replication recordings (Supplementary
# Table S5), GPU half. Same protocol as submit_cross_recording_cpu.sh (100
# iterations, all GPU-capable implementations), on one H100.
#
#   task 0: ds004504 sub-37, 15 components
#   task 1: ds004621 sub-01, 64 components
#
# Result layout: $AMICA_COMPARATOR_RESULTS/<dataset>/gpu/<impl>_sub-NN_seed0_result.json
#
#SBATCH --job-name=jamica_crossrec_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1

REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"
export COMPETITORS_VENV="${COMPETITORS_VENV:-/scratch/$USER/jamica/.venv_competitors/bin/python}"
export PAMICA_VENV="${PAMICA_VENV:-/scratch/$USER/jamica-benchmark/.venv_pamica/bin/python}"
for _v in "$AMICA_PYTHON_VENV" "$COMPETITORS_VENV" "$PAMICA_VENV"; do
    [ -x "$_v" ] || { echo "FATAL: no interpreter at $_v" >&2; exit 1; }
done
"$COMPETITORS_VENV" "$SLURM_SUBMIT_DIR/check_env.py" verify --venv competitors || exit 1
"$PAMICA_VENV"      "$SLURM_SUBMIT_DIR/check_env.py" verify --venv pamica      || exit 1
source "$SLURM_SUBMIT_DIR/assert_jamica.sh" || exit 1

# The fits must run on the GPU in every venv; a silent CPU fallback would still
# produce agreement numbers, just not the ones this table reports.
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
sys.exit(0 if ok else 1)
PYCHK
}
gpu_check "$AMICA_PYTHON_VENV" jamica      || { echo "FATAL: jamica venv cannot see the GPU" >&2; exit 1; }
gpu_check "$COMPETITORS_VENV"  competitors || { echo "FATAL: competitors venv cannot see the GPU" >&2; exit 1; }
gpu_check "$PAMICA_VENV"       pamica      || { echo "FATAL: pamica venv cannot see the GPU" >&2; exit 1; }

case "$SLURM_ARRAY_TASK_ID" in
    0) DS=ds004504; SUBJ=37; NCOMP=15 ;;
    1) DS=ds004621; SUBJ=1;  NCOMP=64 ;;
    *) echo "FATAL: unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

export AMICA_COMPARATOR_RESULTS="${CROSSREC_RESULTS_DIR:-/scratch/$USER/jamica_v030/cross_recording}/$DS"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== cross-recording agreement (GPU): $DS sub-$SUBJ, $NCOMP components, 100 iterations ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
python ../comparator/implementation_perf.py \
    --dataset "$DS" --subject "$SUBJ" --input-level bids \
    --n-components "$NCOMP" --max-iter 100 \
    --amica-device gpu --competitor-device gpu \
    --amica-chunk-size auto --nvml-crosscheck \
    --out-tag gpu \
    --skip amica_python_numpy amica_python_jax

echo "=== DONE $DS. Results under $AMICA_COMPARATOR_RESULTS/gpu/ ==="
