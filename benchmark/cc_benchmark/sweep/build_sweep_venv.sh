#!/bin/bash
# One-time environment build for the block-size sweep clusters (Trillium-GPU and
# Narval), whose compute nodes have no internet. Builds the venv layout fir_env.sh
# expects ($REPO_ROOT/.venv_fir) from the Alliance wheelhouse (--no-index, no
# compilation) plus the pinned jamica release from PyPI, a small pure-Python
# wheel. This is the supported use of pip on a login node: an environment build,
# not compute. No package is imported here; identity is checked from metadata.
#
# Run once on the LOGIN node, from this directory:
#   bash build_sweep_venv.sh                # CPU cluster (Narval)
#   SWEEP_GPU=1 bash build_sweep_venv.sh    # GPU cluster (Trillium-GPU): adds the CUDA plugin wheels
#
# Every sbatch script afterwards just does `source ../fir_env.sh`, which finds the
# venv complete and only runs check_env.py verify.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # benchmark/cc_benchmark/sweep
CC="$(cd "$HERE/.." && pwd)"                              # benchmark/cc_benchmark
REPO_ROOT="$(cd "$CC/../.." && pwd)"
VENV="$REPO_ROOT/.venv_fir"

[ -f "$CC/env.local" ] && source "$CC/env.local"
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ] && \
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023 python/3.11 scipy-stack/2026a
if [ "${SWEEP_GPU:-0}" = "1" ]; then
    module load "${AMICA_CUDA_MODULE:-cuda/12.6}"
    module load "${AMICA_CUDNN_MODULE:-cudnn}" || echo "WARN: no cudnn module; jax ships its own cuDNN wheel" >&2
fi

complete() {
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip show -q jamica jax jaxlib mne mne-bids scikit-learn >/dev/null 2>&1
}

if complete; then
    echo "venv complete at $VENV"
else
    echo "building $VENV"
    rm -rf "$VENV"
    virtualenv --no-download "$VENV"
    source "$VENV/bin/activate"
    pip install --no-index --upgrade pip
    # Pinned scientific stack first (jax/jaxlib/numpy/scipy/mne/mne-bids from pins.toml).
    while read -r s; do
        [ -n "$s" ] && (pip install --no-index "$s" 2>/dev/null || pip install "$s")
    done < <(python "$CC/check_env.py" stack-specs --venv fir)
    # The harness (this repo) with its CPU JAX extra; base deps from the wheelhouse when present.
    pip install --no-index -e "$REPO_ROOT[jax-cpu]" 2>/dev/null || pip install -e "$REPO_ROOT[jax-cpu]"
    if [ "${SWEEP_GPU:-0}" = "1" ]; then
        # jax + jaxlib alone give a CPU-only JAX: without the CUDA plugin packages
        # jax.devices() is [CpuDevice] on a node holding an H100 and the job runs to
        # its wall clock on CPU without ever erroring. Pin the plugin to the jax version.
        JAXV=$(python "$CC/check_env.py" stack-specs --venv fir | sed -n 's/^jax==//p')
        pip install --no-index "jax_cuda12_plugin==$JAXV" "jax_cuda12_pjrt==$JAXV" 2>/dev/null \
            || pip install "jax-cuda12-plugin==$JAXV" "jax-cuda12-pjrt==$JAXV"
    fi
    # The package under test at the pinned PyPI release.
    while read -r spec; do
        [ -n "$spec" ] && pip install "$spec"
    done < <(python "$CC/check_env.py" specs --venv fir)
    python "$CC/check_env.py" lock --venv fir
fi

source "$VENV/bin/activate"
python "$CC/check_env.py" verify --venv fir
pip show jamica jax jaxlib numpy scipy mne 2>/dev/null | grep -E '^(Name|Version)' | paste - - | sed 's/Name: //; s/Version: /==/'
echo "build OK: $VENV (GPU=${SWEEP_GPU:-0}). Import checks run inside the jobs, not here."
