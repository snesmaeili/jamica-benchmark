#!/usr/bin/env bash
# Shared configuration for the local-machine benchmark scripts.
#
# The cluster side has fir_env.sh; this is its counterpart for a workstation or
# laptop. Everything is overridable from the environment, so a collaborator with
# a different layout sets variables rather than editing scripts.
#
# Source it, do not execute it:  source env.sh

# Where the package under test lives. The runner imports `amica` from whichever
# interpreter AMICA_PYTHON_VENV points at, so an editable install of this repo
# is what makes local changes visible to the benchmark.
: "${AMICA_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../amica-python" && pwd)}"
: "${BENCH_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Interpreters. The competitors need their own environment: pAMICA requires
# Python >= 3.12 and a newer torch than the older implementations were built
# against, so it cannot share one with them unless that venv is already 3.12+.
: "${AMICA_PYTHON_VENV:=$AMICA_REPO/.venv-dev/Scripts/python.exe}"
: "${COMPETITORS_VENV:=C:/amica-venvs/comp/Scripts/python.exe}"
: "${PAMICA_VENV:=$COMPETITORS_VENV}"

# MNE's sample dataset, used as the local fixture. Set this explicitly rather
# than relying on MNE's stored config, which goes stale whenever the data moves
# between drives and then fails in a way that looks like a benchmark bug.
: "${MNE_DATASETS_SAMPLE_PATH:=E:/mne_data}"

# Fixture. 30 components x 166,800 samples is the shape the cross-implementation
# comparison uses, so local numbers stay comparable with the published table.
: "${N_COMPONENTS:=30}"
: "${DATASET:=mne_sample}"

export AMICA_REPO BENCH_REPO AMICA_PYTHON_VENV COMPETITORS_VENV PAMICA_VENV
export MNE_DATASETS_SAMPLE_PATH N_COMPONENTS DATASET

PY="$AMICA_PYTHON_VENV"
export PY

if [ ! -x "$PY" ]; then
    echo "env.sh: no interpreter at $PY" >&2
    echo "  set AMICA_PYTHON_VENV to a python with an editable install of amica" >&2
    return 1 2>/dev/null || exit 1
fi
