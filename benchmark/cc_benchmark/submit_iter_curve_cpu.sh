#!/bin/bash
# Cluster CPU: runtime as a function of iteration count, every implementation.
#
# Produces the CPU panel of the runtime-vs-iterations figure. No implementation
# records per-iteration times, and hooking four third-party training loops to add
# them would make each line a different measurement, so the curve is built the
# uniform way: the same fit run to four iteration caps, each timed end to end.
# Every plotted point is a measured fit.
#
# Why all four points are re-run rather than appended to the archived 100- and
# 600-iteration results: jamica now blocks the E-step by default (chunk_size=
# "auto"), so its archived points came from a different implementation. Mixing
# them into one line would draw a curve no single version of the code produces.
# The competitors are unchanged, but they are re-run alongside so that every
# point in the figure comes from one campaign on one node.
#
# This also re-measures peak RSS at 64 components x 785,328 samples, where the
# archived numbers (jamica full batch 11.28 GiB, blocked 6.63 GiB) predate the
# blocking change.
#
# One array task per implementation: a failure costs one line, not the figure,
# and the short implementations do not wait behind the long ones.
#
#SBATCH --job-name=amica_iter_curve_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -o pipefail

cd "$SLURM_SUBMIT_DIR"          # benchmark/cc_benchmark/
source fir_env.sh || exit 1               # modules + .venv_fir + env.local (BIDS_ROOT, AMICA_RESULTS_DIR, ...)

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

# Walltime justification, from the archived 100- and 600-iteration runs on this
# exact problem (per-iteration cost = (t600 - t100) / 500):
#   amica_python_jax          2.73 s/iter -> ~1.7 h for 100+400+700+1000
#   amica_python_jax_chunked  2.92 s/iter -> ~1.8 h   (expected lower now)
#   pamica_torch              3.33 s/iter -> ~2.0 h
#   scott_huberty_torch       3.84 s/iter -> ~2.3 h
#   fortran_amica17           5.96 s/iter -> ~3.6 h
#   pyamica_torch             7.47 s/iter -> ~4.6 h   <- sets the 6 h request
ALL_IMPLS=(amica_python_jax amica_python_jax_chunked pamica_torch \
           scott_huberty_torch pyamica_torch fortran_amica17)
KEEP="${ALL_IMPLS[$SLURM_ARRAY_TASK_ID]}"

# The orchestrator runs every implementation per invocation, so isolating one
# per array task means skipping the complement.
SKIP="amica_python_numpy"
for impl in "${ALL_IMPLS[@]}"; do
    [ "$impl" = "$KEEP" ] && continue
    [ "$impl" = "fortran_amica17" ] && continue     # gated by --include-fortran, not --skip
    SKIP="$SKIP $impl"
done

FORTRAN_OPT=""
if [ "$KEEP" = "fortran_amica17" ]; then
    # Default to the archived reference build, not whatever amica17 is nearest.
    # /scratch/$USER/fortran_parity/amica17_build/amica17 is an ablation build;
    # using it once produced a worst matched |r| of 0.2765 against an archived
    # 0.9391, which reads as a parity failure rather than the wrong binary.
    # Hence the checksum: this row is only meaningful for the reference build.
    # Default to the group-readable staged copy; the expected sha is the SINGLE
    # source of truth from pins.toml (no hard-coded second copy that can drift).
    export AMICA17_BIN="${AMICA17_BIN:-$SLURM_SUBMIT_DIR/../../fortran/amica17}"
    AMICA17_SHA_EXPECTED="${AMICA17_SHA_EXPECTED:-$("$AMICA_PYTHON_VENV" "$SLURM_SUBMIT_DIR/check_env.py" fortran-sha)}"
    export AMICA17_SHA_EXPECTED          # run_fortran.py also asserts it per fit
    if [ -x "${AMICA17_BIN}" ]; then
        _sha=$(sha256sum "$AMICA17_BIN" | awk '{print $1}')
        if [ "$_sha" != "$AMICA17_SHA_EXPECTED" ]; then
            echo "FATAL: $AMICA17_BIN has sha $_sha, expected $AMICA17_SHA_EXPECTED (pins.toml)" >&2
            exit 1
        fi
        echo "Fortran reference binary verified: $AMICA17_BIN (sha ${_sha:0:12}…)"
        module load openmpi/4.1.5 flexiblas 2>/dev/null || true
        export GNU_TIME_BIN="${GNU_TIME_BIN:-/usr/bin/time}"
        FORTRAN_OPT="--include-fortran"
    else
        echo "FATAL: no Fortran binary at $AMICA17_BIN" >&2
        exit 1
    fi
fi

# A distinct tag per component count, so a 30-component campaign cannot land on
# top of a 64-component one. The two are not interchangeable: per-iteration cost
# is not linear in the component count (measured 3.92/6.83/17.45/19.87 ms/iter at
# C=16/32/48/64 at fixed sample count, where a linear fit misses by up to 46%), so
# mixing them in one directory would silently blend two scalings into one curve.
# Running the laptop fixture (mne_sample) here makes the cluster and local
# panels differ in hardware alone, instead of in dataset, rank and recording
# length at once. MNE_DATASETS_SAMPLE_PATH is set explicitly rather than read
# from MNE's stored config, which goes stale whenever data moves and then
# fails in a way that reads as a benchmark bug rather than a path problem.
export MNE_DATASETS_SAMPLE_PATH="${MNE_DATASETS_SAMPLE_PATH:-$HOME/mne_data}"
export AMICA_ITER_TAG="${AMICA_ITER_TAG:-itercurve_cpu}"
export AMICA_COMPARATOR_RESULTS="${AMICA_MEM_RESULTS:-${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}}/itercurve/${AMICA_ITER_TAG}"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== task $SLURM_ARRAY_TASK_ID: $KEEP (skipping: $SKIP) ==="
python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"

# Overridable so a single missing point can be refilled without re-running the
# whole campaign, which at 2200 iterations per implementation is hours.
ITERS="${AMICA_ITERS:-100 400 700 1000}"
for IT in $ITERS; do
    echo "--- $KEEP @ max_iter=$IT ---"
    python ../comparator/implementation_perf.py \
        --dataset "${AMICA_MEM_DATASET:-ds004505}" \
        --subject "${AMICA_MEM_SUBJECT:-1}" \
        --input-level "${AMICA_INPUT_LEVEL:-bids}" \
        --n-components "${AMICA_MEM_NCOMP:-64}" \
        --max-iter "$IT" \
        --amica-device cpu --competitor-device cpu \
        --amica-chunk-size "${AMICA_MEM_CHUNK:-auto}" \
        $FORTRAN_OPT \
        --out-tag "${AMICA_ITER_TAG}/iter${IT}" \
        --skip $SKIP
done

echo "=== DONE ($KEEP, C=${AMICA_MEM_NCOMP:-64}). Results under $AMICA_COMPARATOR_RESULTS/ ==="
