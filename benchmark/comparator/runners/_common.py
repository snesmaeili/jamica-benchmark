"""Shared runner protocol for the three-implementation perf comparison.

Each runner script:
  - Takes --input (path to .npz with key 'X' = (n_components, n_samples)).
  - Takes --output (path to write the result JSON).
  - Takes --config (JSON string with at minimum: max_iter, n_mix, lrate, seed).
  - Writes a JSON dict with the keys defined in `RESULT_KEYS` below.

Peak-RSS is the process high-water mark: resource.getrusage(RUSAGE_SELF).ru_maxrss
on POSIX (the TRUE peak — captures a transient high that occurred mid-fit, not just
the instantaneous RSS at call time) and psutil peak_wset on Windows. Each runner
records a pre-fit baseline (baseline_rss_gb) right before fit() and reports
delta_rss_gb = peak - baseline (the fit's marginal footprint) next to the absolute
peak. peak_vram_gb is the GPU device peak when on GPU (jax peak_bytes_in_use /
torch max_memory_allocated), else None. All values are GiB (binary, /1024**n) to
match the in-tree benchmark.runner convention.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import psutil

RESULT_KEYS = (
    "implementation", "n_components", "n_samples", "max_iter",
    "fit_time_s", "peak_rss_gb", "baseline_rss_gb", "delta_rss_gb",
    "peak_vram_gb", "nvml_peak_vram_gb",
    "ll_final", "ll_history", "W",
    "device", "dtype", "n_iter",
    # Written automatically by write_result(); see stack_provenance().
    "provenance",
)

# Distributions whose *identity* (version + VCS commit) we care about — the
# implementations under comparison. Only those actually installed in the runner's
# venv end up in the block, so this one list is safe to probe from every runner.
# (scott-huberty and Sina's amica both install under the "amica" dist name, but
# they live in separate venvs, so whichever is present is the right one.)
_PROVENANCE_DISTS = ("pamica", "pyamica", "amica", "amica_python", "pyAMICA")
# Numerical stack: recorded so pamica-vs-pyamica time/memory deltas can be
# decomposed from torch/jax/numpy version deltas (the two venvs carry different
# torch builds). importlib.metadata.version() reports the installed version
# without importing the (heavy) package, so probing jax from the torch venv is
# cheap and simply absent.
_PROVENANCE_STACK = ("torch", "jax", "jaxlib", "numpy", "scipy", "mne")


def _canonical_dist_key(name: str) -> str:
    """PEP 503 normalized name: lowercase, runs of [-_.] collapsed to one '-'.

    So "pyamica", "pyAMICA", "py_amica" and "py.amica" all map to "pyamica" —
    the same rule pip uses to decide two dists are the same project.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _dist_vcs(name: str) -> dict:
    """PEP 610 source identity for a dist installed from git: {url, commit}.

    pip records a `git+…` install's origin in the distribution's
    `direct_url.json`: the source `url` (e.g. the github.com/owner/repo link,
    which disambiguates which project a bare import name like `amica` came from)
    and the resolved `vcs_info.commit_id`. A PyPI/wheel install has neither.
    Returns only the keys that are present (empty dict for a non-VCS install).
    """
    try:
        raw = importlib_metadata.distribution(name).read_text("direct_url.json")
        if not raw:
            return {}
        info = json.loads(raw)
        out: dict = {}
        url = info.get("url")
        if url:
            # Strip pip's "git+" scheme prefix so the value reads as the plain
            # repository URL (the resolved rev lives in vcs_info, recorded below).
            out["url"] = url[4:] if url.startswith("git+") else url
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if commit:
            out["commit"] = commit
        return out
    except Exception:
        return {}


def stack_provenance(*dist_names: str) -> dict:
    """Self-describing build/env block for a result JSON.

    For each installed distribution in `dist_names` (default: every competitor
    implementation) records its `importlib.metadata.version()` and, for git
    installs, the PEP 610 `direct_url.json` commit SHA. Also records the Python
    version, interpreter path, and the numerical-stack versions
    (torch/jax/jaxlib/numpy/scipy/mne) that are installed. Wired into
    write_result() so every runner emits it uniformly and for free.

    Distributions are keyed by their canonical metadata name and de-duplicated:
    importlib.metadata normalizes names (PEP 503), so probing both "pyamica" and
    "pyAMICA" would otherwise report the same installed dist twice.
    """
    names = dist_names or _PROVENANCE_DISTS
    packages: dict = {}
    seen: set = set()
    for name in names:
        # The whole per-name body is guarded: provenance must never be able to
        # crash a result write that would otherwise have succeeded, even on a
        # distribution with malformed/missing metadata.
        try:
            dist = importlib_metadata.distribution(name)
            canonical = (dist.metadata["Name"] or name)
            norm = _canonical_dist_key(canonical)
            if norm in seen:
                continue  # same dist reached via a differently-cased probe name
            seen.add(norm)
            entry: dict = {"version": dist.version}
            entry.update(_dist_vcs(name))  # url + commit, when git-installed
            packages[canonical] = entry
        except Exception:
            continue  # not installed, or unreadable metadata
    stack: dict = {}
    for name in _PROVENANCE_STACK:
        try:
            stack[name] = importlib_metadata.version(name)
        except Exception:
            pass  # not importable in this venv
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "stack": stack,
    }


def parse_runner_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npz with key X (n_components, n_samples)")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--config", required=True, help="JSON-encoded config dict")
    args = parser.parse_args()
    cfg = json.loads(args.config)
    return args, cfg


def load_data(input_path: str) -> np.ndarray:
    z = np.load(input_path)
    # asarray (not .astype) avoids a needless full copy when X is already float64 —
    # that copy otherwise creates a transient np.load RSS spike that, via the
    # monotonic high-water mark, inflates the pre-fit baseline for every impl.
    X = np.asarray(z["X"], dtype=np.float64)
    return X  # (n_components, n_samples)


def peak_rss_gb() -> float:
    """Process peak resident-set size (high-water mark), in GiB.

    POSIX: resource.getrusage(RUSAGE_SELF).ru_maxrss — the kernel's TRUE
    high-water mark (KiB on Linux, bytes on macOS), so a transient peak that
    occurred mid-fit is captured, unlike the instantaneous psutil rss the old
    code returned on Linux. Windows: psutil peak_wset (also a high-water mark).
    Binary GiB (/1024**n) to match amica_python.benchmark.runner._measure_peak_memory.
    """
    if sys.platform != "win32":
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            div = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
            return float(ru) / div
        except Exception:
            pass
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows high-water mark (bytes)
        return info.peak_wset / 1024 ** 3
    return info.rss / 1024 ** 3


def baseline_rss_gb() -> float:
    """Pre-fit RSS baseline (GiB) — same high-water source as peak_rss_gb().

    Call right before fit() so the value is the post-import / pre-fit floor that
    delta_rss_gb() subtracts off. (peak_rss_gb is monotonic, so this is the
    high-water-so-far = interpreter + data + framework import.)
    """
    return peak_rss_gb()


def delta_rss_gb(baseline: float) -> float:
    """Peak RSS attributable to the fit: current peak minus the pre-fit baseline (GiB)."""
    return peak_rss_gb() - baseline


def start_nvml_sampler(enabled: bool, gpu_index: int = 0, interval_s: float = 0.05):
    """Start a background sampler of whole-GPU 'used' VRAM via NVML; return a handle.

    Framework-neutral cross-check: on a DEDICATED GPU (our `--gres=gpu:h100:1` case)
    the whole-GPU used peak = our process's total device footprint = allocator tensors
    (peak_bytes_in_use / max_memory_allocated) PLUS the fixed CUDA-context floor
    (cuDNN/cuBLAS, few-hundred-MB) that the per-framework allocator counters omit.
    Returns None if disabled or pynvml/GPU unavailable (caller treats as no cross-check).
    Pass the handle to stop_nvml_sampler() to read the peak. Requires `pip install
    nvidia-ml-py` in the venv; silently degrades to None otherwise.
    """
    if not enabled:
        return None
    try:
        import threading
        import pynvml
        pynvml.nvmlInit()
        handle_dev = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        state = {"stop": threading.Event(), "peak": 0.0, "nvml": pynvml, "thread": None}

        def _loop():
            while not state["stop"].is_set():
                try:
                    used = pynvml.nvmlDeviceGetMemoryInfo(handle_dev).used
                    g = float(used) / 1024 ** 3
                    if g > state["peak"]:
                        state["peak"] = g
                except Exception:
                    pass
                state["stop"].wait(interval_s)

        state["thread"] = threading.Thread(target=_loop, daemon=True)
        state["thread"].start()
        return state
    except Exception:
        return None


def stop_nvml_sampler(handle) -> float | None:
    """Stop the sampler and return peak whole-GPU used VRAM in GiB (or None)."""
    if not handle:
        return None
    try:
        handle["stop"].set()
        handle["thread"].join(timeout=1.0)
        handle["nvml"].nvmlShutdown()
        return float(handle["peak"])
    except Exception:
        return None


def write_result(output_path: str, result: dict) -> None:
    # Version-stamp every result at this single choke point so all six runners
    # emit a uniform, auditable provenance block for free. setdefault() lets a
    # runner override it deliberately, but nothing does today.
    result.setdefault("provenance", stack_provenance())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    if "error" in result:
        print(f"[{result.get('implementation', '?')}] ERROR: {result['error']}  -> {output_path}")
        return
    vram = result.get("peak_vram_gb")
    vram_s = f"  vram={vram:.2f}GB" if vram is not None else ""
    print(f"[{result['implementation']}] {result['fit_time_s']:.2f}s  "
          f"peak={result['peak_rss_gb']:.2f}GB  "
          f"delta={result.get('delta_rss_gb', float('nan')):.2f}GB{vram_s}  "
          f"ll={result['ll_final']:.4f}  -> {output_path}")
