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
# 600-iteration results: amica now blocks the E-step by default (chunk_size=
# "auto"), so its archived points came from a different implementation. Mixing
# them into one line would draw a curve no single version of the code produces.
# The competitors are unchanged, but they are re-run alongside so that every
# point in the figure comes from one campaign on one node.
#
# This also re-measures peak RSS at 64 components x 785,328 samples, where the
# archived numbers (amica full batch 11.28 GiB, blocked 6.63 GiB) predate the
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
source fir_env.sh               # modules + .venv_fir + env.local (BIDS_ROOT, AMICA_RESULTS_DIR, ...)

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
    if [ -n "${AMICA17_BIN:-}" ] && [ -x "${AMICA17_BIN}" ]; then
        module load openmpi/4.1.5 flexiblas 2>/dev/null || true
        export GNU_TIME_BIN="${GNU_TIME_BIN:-/usr/bin/time}"
        FORTRAN_OPT="--include-fortran"
    else
        echo "AMICA17_BIN unset or not executable -- nothing to do for this task."
        exit 0
    fi
fi

export AMICA_COMPARATOR_RESULTS="${AMICA_RESULTS_DIR:-/scratch/$USER/amica_mem}/itercurve/cpu"
mkdir -p "$AMICA_COMPARATOR_RESULTS"

echo "=== task $SLURM_ARRAY_TASK_ID: $KEEP (skipping: $SKIP) ==="
python -c "import sys; print('python:', sys.executable, sys.version.split()[0])"

for IT in 100 400 700 1000; do
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
        --out-tag "itercurve_cpu/iter${IT}" \
        --skip $SKIP
done

echo "=== DONE ($KEEP). Results under $AMICA_COMPARATOR_RESULTS/itercurve_cpu/ ==="
