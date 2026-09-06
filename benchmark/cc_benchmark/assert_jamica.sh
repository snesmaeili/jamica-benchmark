#!/bin/bash
# Assert WHICH jamica build a comparator job measures, and print its identity
# into the job log. Sourced (not executed) by the submit scripts that launch
# benchmark/comparator/implementation_perf.py, after fir_env.sh, with
# AMICA_PYTHON_VENV pointing at the interpreter that runs run_amica_python.py.
#
#   AMICA_SRC unset  -> the release installed in $AMICA_PYTHON_VENV must be the
#                       version pinned in pins.toml ([[venv]] fir, package jamica),
#                       imported from inside that venv, with the chunked E-step
#                       default (chunk_size="auto").
#   AMICA_SRC set    -> a source checkout on PYTHONPATH is measured instead. Its
#                       HEAD must equal the commit pinned in pins.toml unless
#                       AMICA_ALLOW_SRC_DRIFT=1, and the worktree must be clean.
#
# Runs on the compute node (importing jamica imports jax). Returns non-zero on
# any mismatch; the caller does `source assert_jamica.sh || exit 1`.
_aj_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_aj_py="${AMICA_PYTHON_VENV:?assert_jamica.sh: AMICA_PYTHON_VENV must be set}"
_aj_fail() { echo "assert_jamica: $*" >&2; return 1; }

if [ -z "${AMICA_SRC:-}" ]; then
    _aj_want_ver=$("$_aj_py" "$_aj_dir/check_env.py" pin --venv fir --name jamica --field version) \
        || _aj_fail "pins.toml has no version pin for jamica in venv fir" || return 1
    AMICA_WANT_VERSION="$_aj_want_ver" "$_aj_py" - <<'PYCHECK' || return 1
import os, sys
import jamica
from jamica import AmicaConfig
want = os.environ["AMICA_WANT_VERSION"]
src = os.path.realpath(jamica.__file__)
prefix = os.path.realpath(sys.prefix)
if jamica.__version__ != want:
    sys.exit(f"FATAL: jamica {jamica.__version__} imported, pins.toml pins {want}")
if not src.startswith(prefix):
    sys.exit(f"FATAL: jamica imported from {src}, outside the venv {prefix} (PYTHONPATH shadowing?)")
if AmicaConfig().chunk_size != "auto":
    sys.exit(f"FATAL: chunk_size default is {AmicaConfig().chunk_size!r}, not 'auto'")
print(f"jamica OK: {jamica.__version__} from {src} | default chunk_size={AmicaConfig().chunk_size!r}")
PYCHECK
    echo "jamica pin OK: release ${_aj_want_ver} (installed wheel; AMICA_SRC unset)"
else
    # Source checkout: it must be what the runner imports, and it must be the
    # pinned commit unless drift is allowed on purpose.
    AMICA_SRC="$AMICA_SRC" PYTHONPATH="$AMICA_SRC${PYTHONPATH:+:$PYTHONPATH}" "$_aj_py" - <<'PYCHECK' || return 1
import os, sys
import jamica
from jamica import AmicaConfig
src = os.path.realpath(jamica.__file__)
want = os.path.realpath(os.environ["AMICA_SRC"])
if not src.startswith(want):
    sys.exit(f"FATAL: imported jamica from {src}, expected under {want}")
if AmicaConfig().chunk_size != "auto":
    sys.exit("FATAL: this build predates E-step blocking (chunk_size default is not 'auto')")
print(f"jamica OK (source checkout): {src} | default chunk_size={AmicaConfig().chunk_size!r}")
PYCHECK
    if [ "${AMICA_ALLOW_SRC_DRIFT:-0}" != "1" ]; then
        _aj_want=$("$_aj_py" "$_aj_dir/check_env.py" pin --venv fir --name jamica) || return 1
        _aj_got=$(git -C "$AMICA_SRC" rev-parse HEAD 2>/dev/null)
        if [ "$_aj_got" != "$_aj_want" ]; then
            _aj_fail "AMICA_SRC HEAD ${_aj_got:-<none>} != pinned jamica ${_aj_want} (set AMICA_ALLOW_SRC_DRIFT=1 to measure a non-pinned checkout on purpose)" || return 1
        fi
        if ! git -C "$AMICA_SRC" diff --quiet HEAD 2>/dev/null; then
            _aj_fail "AMICA_SRC has uncommitted changes (dirty worktree at pinned HEAD; set AMICA_ALLOW_SRC_DRIFT=1 to measure it on purpose)" || return 1
        fi
        echo "jamica pin OK: AMICA_SRC HEAD == ${_aj_want} (clean)"
    fi
    # Record which commit produced these numbers: a `git pull` in that checkout
    # silently changes what a later job measures, so the SHA goes in the log.
    echo "jamica source commit: $(git -C "$AMICA_SRC" rev-parse --short HEAD 2>/dev/null) $(git -C "$AMICA_SRC" log -1 --format=%s 2>/dev/null)"
fi
unset _aj_dir _aj_py _aj_want_ver _aj_want _aj_got
