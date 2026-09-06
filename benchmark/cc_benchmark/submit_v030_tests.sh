#!/bin/bash
# Supplementary Table S8 rows that are tests rather than benchmarks, run against
# the jamica release installed by fir_env.sh:
#   (1) the release's own MNE interoperability suite (tests/test_mne_integration.py,
#       tests/test_mne_real.py, tests/test_amica_ica.py), importing the venv wheel;
#   (2) the full-batch vs chunked pair on the MNE sample fixture
#       (scripts/performance/validate_chunking_mne.py with AMICA_USE_RELEASE=1).
# Nothing is asserted about published values here: the JUnit XML and summary.json
# are compared off-cluster.
#
# The wheel does not ship the tests, so a checkout of the jamica repository at the
# pinned tag is needed (login node, one-time; git only):
#   git clone --branch v0.3.0 --depth 1 https://github.com/snesmaeili/jamica.git \
#       /scratch/$USER/jamica_v030/jamica-src
#
#SBATCH --job-name=jamica_v030_tests
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1
REPO_ROOT="${REPO_ROOT:-$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)}"
export AMICA_PYTHON_VENV="${AMICA_PYTHON_VENV:-$REPO_ROOT/.venv_fir/bin/python}"
source assert_jamica.sh || exit 1

OUT="${V030_TESTS_DIR:-/scratch/$USER/jamica_v030/tests}/job_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
export MNE_DATA="${MNE_DATA:-/scratch/$USER/mne_data}"
mkdir -p "$MNE_DATA"

# Test-only dependencies (compute node; fir compute nodes reach PyPI). h5io backs
# AmicaICA.save(); onnxruntime is the backend mne-icalabel needs at run time.
for pair in pytest:pytest pytest_timeout:pytest-timeout h5io:h5io mne_icalabel:mne-icalabel onnxruntime:onnxruntime; do
    mod="${pair%%:*}"; spec="${pair##*:}"
    python -c "import $mod" 2>/dev/null \
        || pip install --no-index --quiet "$spec" 2>/dev/null \
        || pip install --quiet "$spec" \
        || { echo "FATAL: cannot install $spec" >&2; exit 1; }
done

# --- 1. MNE interoperability suite ----------------------------------------------
JAMICA_SRC="${JAMICA_SRC:-/scratch/$USER/jamica_v030/jamica-src}"
WANT=$(python check_env.py pin --venv fir --name jamica) || exit 1
GOT=$(git -C "$JAMICA_SRC" rev-parse HEAD 2>/dev/null)
if [ "$GOT" != "$WANT" ]; then
    echo "FATAL: $JAMICA_SRC is at ${GOT:-<none>}; the pinned release commit is $WANT" >&2
    exit 1
fi
# Copy the suite and the pytest configuration away from the source package, so
# `import jamica` inside the tests resolves to the installed wheel and not to
# the checkout sitting next to them.
SUITE="$OUT/suite"
rm -rf "$SUITE"; mkdir -p "$SUITE"
cp -r "$JAMICA_SRC/tests" "$SUITE/tests"
cp "$JAMICA_SRC/pyproject.toml" "$SUITE/pyproject.toml"
( cd "$SUITE" && python - <<'PY'
import sys, jamica
ok = jamica.__file__.startswith(sys.prefix)
print("suite imports jamica", jamica.__version__, "from", jamica.__file__, "(venv)" if ok else "(NOT the venv)")
sys.exit(0 if ok else 1)
PY
) || { echo "FATAL: the suite would import jamica from outside the venv" >&2; exit 1; }

echo "=== MNE interoperability tests ($JAMICA_SRC @ ${GOT:0:10}) ==="
( cd "$SUITE" && python -m pytest tests -k "mne or interop or amica_ica" -p no:cacheprovider \
      --junitxml="$OUT/mne_interop.xml" ) > "$OUT/mne_interop.log" 2>&1 \
    || echo "pytest reported failures (see $OUT/mne_interop.log)"
tail -5 "$OUT/mne_interop.log"
# Guard against the vacuous pass: a -k expression that selects nothing exits 0.
python - "$OUT/mne_interop.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    n = int(suite.get("tests", 0))
    bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    print(f"collected {n} test(s), {bad} failing/erroring, {skipped} skipped")
    if n == 0:
        print("WARNING: zero tests collected -- this proves nothing.")
except Exception as e:
    print(f"WARNING: could not read junit xml: {e}")
PY

# --- 2. full-batch vs chunked, published fixture -------------------------------
echo "=== full-batch vs chunked (MNE sample, 1-40 Hz, average reference, 30 PCs, 100 iterations) ==="
export AMICA_USE_RELEASE=1
export AMICA_CHUNKING_OUT="$OUT/chunking"
( cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python scripts/performance/validate_chunking_mne.py ) \
    || echo "chunking script FAILED"
echo "--- summary.json ---"
cat "$OUT/chunking/summary.json" 2>/dev/null || echo "WARNING: no summary.json produced"

echo
echo "=== outputs ==="
ls -la "$OUT"
echo "DONE"
