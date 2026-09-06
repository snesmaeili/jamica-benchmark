#!/bin/bash
#SBATCH --job-name=jamica_smoke_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# GPU smoke gate for the campaign arrays: the shared venv must expose the H100
# to JAX, a short JAX-GPU fit must complete, and the result JSON must carry the
# pinned jamica release in its provenance block. Runs after submit_smoke_mne.sh
# (which builds the venv) and gates the GPU arrays via --dependency=afterok.
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1     # refuses a CPU-only venv on a GPU job

export AMICA_RESULTS_DIR="${AMICA_RESULTS_DIR:-/scratch/$USER/jamica_v030/validation_v3}/smoke_gpu"
mkdir -p "$AMICA_RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader || true

python - <<'PY' || { echo "SMOKE FAIL: JAX does not see the GPU" >&2; exit 1; }
import sys
import jax, jamica
devs = jax.devices()
print("jax", jax.__version__, "devices", devs, "| jamica", jamica.__version__, jamica.__file__)
sys.exit(0 if any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in devs) else 1)
PY

python run_one_subject.py --subject 1 --dataset mne --backend jax --device gpu \
    --n-iter 50 --schema-version v3 --output-dir "$AMICA_RESULTS_DIR" || exit 1

export AMICA_PINNED_VERSION="$(python check_env.py pin --venv fir --name jamica --field version)"
python - <<'PY'
import glob, json, os
d = os.environ["AMICA_RESULTS_DIR"]
js = sorted(glob.glob(os.path.join(d, "benchmark_sub-*hp*.json")))
assert js, f"SMOKE FAIL: no benchmark JSON written under {d}"
doc = json.load(open(js[-1]))
assert doc.get("_schema_version") == "3.0", f"SMOKE FAIL: schema {doc.get('_schema_version')!r}"
pkgs = ((doc.get("_run") or {}).get("provenance") or {}).get("packages") or {}
got = (pkgs.get("jamica") or {}).get("version")
want = os.environ["AMICA_PINNED_VERSION"]
assert got == want, f"SMOKE FAIL: result provenance says jamica {got!r}, pins.toml pins {want!r}"
am = doc.get("amica") or {}
print("SMOKE GPU OK:", js[-1], "| jamica", got, "| backend", am.get("backend"), "device", am.get("device"),
      "| duration_s", am.get("duration_s") or am.get("fit_time_s"))
PY
