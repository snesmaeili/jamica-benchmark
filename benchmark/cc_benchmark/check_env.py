#!/usr/bin/env python
"""Make "installed == intended" machine-checkable for the benchmark venvs.

`pins.toml` (next to this file) is the single source of truth for the exact git
commit of every AMICA implementation, grouped by venv. Run this tool WITH THE
TARGET VENV'S PYTHON:

  check_env.py specs  --venv NAME   print `git+URL@COMMIT` lines for pip install
  check_env.py verify --venv NAME   assert each pinned dist is installed at the
                                    pinned commit; exit nonzero on any mismatch
                                    or missing dist. Call before the first fit.
  check_env.py lock   --venv NAME   write locks/NAME.lock.json — the as-built
                                    record: each dist's version + commit plus the
                                    resolved torch/jax/jaxlib/numpy/scipy/mne.

`verify` matches installed distributions by their PEP 610 `direct_url.json`
SOURCE URL, not by distribution name — so it is robust to name-canonicalization
collisions (e.g. `pyamica` vs `pyAMICA`) and reports the true installed commit.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata as im
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_DEFAULT = HERE / "pins.toml"
LOCKS_DIR = HERE / "locks"

# Recorded so time/memory deltas can be separated from stack-version deltas.
STACK = ("torch", "jax", "jaxlib", "numpy", "scipy", "mne", "mne-bids")


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - all benchmark venvs are >=3.11
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_venv(pins_path: Path, venv_name: str) -> dict:
    doc = _load_toml(pins_path)
    for venv in doc.get("venv", []):
        if venv.get("name") == venv_name:
            return venv
    names = ", ".join(v.get("name", "?") for v in doc.get("venv", []))
    raise SystemExit(f"check_env: unknown venv {venv_name!r}; pins.toml has: {names}")


def canon_url(url: str) -> str:
    """Normalize a git URL for comparison: drop pip's git+ prefix, .git, trailing /."""
    u = (url or "").strip()
    if u.startswith("git+"):
        u = u[4:]
    u = u.split("@", 1)[0]  # drop any @rev
    u = u.rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    return u.lower()


def _direct_url(dist) -> dict:
    try:
        raw = dist.read_text("direct_url.json")
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def installed_by_url() -> dict:
    """Map canonical source URL -> {name, version, commit} for every git install."""
    out: dict = {}
    for dist in im.distributions():
        info = _direct_url(dist)
        url = info.get("url")
        if not url:
            continue
        commit = (info.get("vcs_info") or {}).get("commit_id")
        out[canon_url(url)] = {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "commit": commit,
        }
    return out


def cmd_specs(venv: dict) -> int:
    for pkg in venv.get("packages", []):
        print(f"git+{pkg['url']}@{pkg['commit']}")
    return 0


def cmd_verify(venv: dict) -> int:
    installed = installed_by_url()
    ok = True
    for pkg in venv.get("packages", []):
        want_url = canon_url(pkg["url"])
        want_commit = pkg["commit"]
        got = installed.get(want_url)
        if got is None:
            ok = False
            print(f"MISSING  {pkg['name']}: no install from {pkg['url']} "
                  f"(is it uninstalled or clobbered by a name collision?)")
            continue
        if got["commit"] != want_commit:
            ok = False
            print(f"MISMATCH {pkg['name']}: installed {got['commit']} "
                  f"!= pinned {want_commit}")
            continue
        print(f"OK       {pkg['name']}: {want_commit} ({got['version']})")
    if not ok:
        print("check_env: installed != intended — refusing to proceed.", file=sys.stderr)
        return 1
    _report_stack_drift(venv)
    return 0


def _stack_versions() -> dict:
    out = {}
    for name in STACK:
        try:
            out[name] = im.version(name)
        except Exception:
            pass
    return out


def _lock_path(venv: dict) -> Path:
    return LOCKS_DIR / f"{venv['name']}.lock.json"


def _report_stack_drift(venv: dict) -> None:
    """Warn (do not fail) if the numerical stack drifted from the recorded lock."""
    lock_path = _lock_path(venv)
    if not lock_path.exists():
        return
    try:
        recorded = json.loads(lock_path.read_text()).get("stack", {})
    except Exception:
        return
    now = _stack_versions()
    for name, was in recorded.items():
        is_now = now.get(name)
        if is_now is not None and is_now != was:
            print(f"WARN     stack drift: {name} {was} (locked) -> {is_now} (installed)",
                  file=sys.stderr)


def cmd_lock(venv: dict) -> int:
    installed = installed_by_url()
    packages = {}
    for pkg in venv.get("packages", []):
        got = installed.get(canon_url(pkg["url"]))
        if got is not None:
            packages[pkg["name"]] = {"version": got["version"], "commit": got["commit"],
                                     "url": pkg["url"]}
    lock = {
        "venv": venv["name"],
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "stack": _stack_versions(),
    }
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    out = _lock_path(venv)
    out.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"check_env: wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("specs", "verify", "lock"))
    parser.add_argument("--venv", required=True,
                        help="venv name in pins.toml (competitors/neuromechanist/pamica/fir)")
    parser.add_argument("--pins", type=Path, default=PINS_DEFAULT)
    args = parser.parse_args(argv)
    venv = load_venv(args.pins, args.venv)
    return {"specs": cmd_specs, "verify": cmd_verify, "lock": cmd_lock}[args.command](venv)


if __name__ == "__main__":
    raise SystemExit(main())
