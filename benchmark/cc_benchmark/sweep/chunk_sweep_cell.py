#!/usr/bin/env python3
"""Block-size / iteration-budget sweep for jamica (the jamica rows of Figures 5, 6
and Supplementary Figure S5).

One invocation = one subject on one device. The recording is preprocessed once
with the comparator pipeline (BIDS load, analysis window / resample, 1-100 Hz plus
mains notch, PCA to --n-components, unit variance per component), the projection
is cached, and every requested (chunk, max_iter) cell is then fitted in a fresh
subprocess through the comparator runner
(benchmark/comparator/runners/run_amica_python.py) with early stopping disabled,
so each cell runs its full iteration budget and wall time is per-iteration
comparable. The other implementations' rows of the published sweep are reused,
so only the two jamica orchestrator keys are fitted here:

    amica_python_jax_chunked   AMICA_CHUNK_SIZE=<chunk>          -> "jamica"
    amica_python_jax           chunk_size=None (full batch)      -> "jamica_fullbatch"

Layout (matching the per-cell result JSONs the earlier campaign aggregated):
    <out-root>/inputs/input_sub-NN.npz                       cached projection
    <out-root>/<device>/c<chunk>_i<iter>_r<rep>/<key>_sub-NN_seed<seed>_result.json
    <out-root>/<device>/manifest_sub-NN.json                 cells, env, provenance

Run from benchmark/cc_benchmark/sweep/ inside an allocation (see the submit_*.sh
scripts); fir_env.sh must have been sourced so BIDS_ROOT_DS4505 and the venv
resolve. Aggregate afterwards with aggregate_chunk_sweep.py, off-cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
COMPARATOR = REPO_ROOT / "benchmark" / "comparator"
for p in (str(REPO_ROOT), str(COMPARATOR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import implementation_perf as ip  # noqa: E402  (benchmark/comparator/implementation_perf.py)

FULL = "full"


def _parse_chunks(spec: str) -> list[int | str]:
    out: list[int | str] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(FULL if tok in (FULL, "fullbatch", "none") else int(tok))
    return out


def _parse_ints(spec: str) -> list[int]:
    return [int(t) for t in spec.split(",") if t.strip()]


def _venv_identity(py: Path) -> dict:
    """Version + file of the jamica the runner subprocess will import (no AMICA_SRC here)."""
    code = ("import json, jamica, jax; print(json.dumps({'jamica_version': jamica.__version__, "
            "'jamica_file': jamica.__file__, 'jax_version': jax.__version__, "
            "'jax_devices': [str(d) for d in jax.devices()]}))")
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("SWEEP_JAX_PLATFORMS", "cpu"))
    cp = subprocess.run([str(py), "-c", code], capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        return {"error": cp.stderr[-500:]}
    return json.loads(cp.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ds004505", choices=["ds004505", "ds004504", "ds004621", "mne_sample"])
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--chunks", default="1024,4096,16384,65536,262144,524288,1048576,full",
                    help="comma list of chunk sizes; 'full' = full-batch key (chunk_size=None)")
    ap.add_argument("--max-iter", type=int, required=True, help="iteration budget of the chunk sweep")
    ap.add_argument("--ladder-iters", default="", help="extra budgets at --ladder-chunk (comma list)")
    ap.add_argument("--ladder-chunk", type=int, default=65536)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--lrate", type=float, default=0.1, help="orchestrator protocol value (implementation_perf default)")
    ap.add_argument("--input-level", default="bids", choices=["bids", "merged"])
    ap.add_argument("--resample-sfreq", type=float, default=250.0)
    ap.add_argument("--out-root", type=Path,
                    default=Path(os.environ.get("SWEEP_RESULTS_DIR", f"/scratch/{os.environ.get('USER', 'user')}/jamica_v030/sweep")))
    ap.add_argument("--cell-timeout", type=float, default=None,
                    help="seconds per cell (default 4 h on GPU, 6 h on CPU)")
    ap.add_argument("--skip-existing", action="store_true", help="skip cells whose result JSON exists without error")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks = _parse_chunks(args.chunks)
    ladder = _parse_ints(args.ladder_iters)
    timeout = args.cell_timeout or (4 * 3600.0 if args.device == "gpu" else 6 * 3600.0)
    py = ip.VENV_AMICA
    runner = ip.RUNNERS_DIR / "run_amica_python.py"
    if not py.exists():
        sys.exit(f"FATAL: jamica venv python missing at {py} (AMICA_PYTHON_VENV)")
    if os.environ.get("AMICA_SRC"):
        sys.exit("FATAL: AMICA_SRC is set; this sweep measures the installed release only")

    subject_tag = f"sub-{args.subject:02d}" if args.dataset != "mne_sample" else "mne_sample"
    dev_dir = args.out_root / args.device
    dev_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.out_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    input_path = inputs_dir / f"input_{subject_tag}.npz"

    # 1. Cached projection (identical to implementation_perf.py's preprocessing).
    if input_path.exists():
        with np.load(input_path) as z:
            X = z["X"]
            meta = {k: z[k].item() for k in z.files if k != "X"}
        print(f"[sweep] cached input {input_path} X={X.shape}")
    else:
        t0 = time.perf_counter()
        if args.dataset == "mne_sample":
            X, meta = ip.preprocess_mne_sample(n_components=args.n_components, seed=args.seed)
        else:
            X, meta = ip.preprocess_bids_subject(
                dataset=args.dataset, subject_id=args.subject, n_components=args.n_components,
                duration_sec=None, resample_sfreq=args.resample_sfreq, seed=args.seed,
                input_level=args.input_level)
        np.savez(input_path, X=X, **{k: v for k, v in meta.items() if isinstance(v, (int, float))})
        print(f"[sweep] preprocessed {subject_tag} X={X.shape} in {time.perf_counter() - t0:.0f}s -> {input_path}")
    n_samples = int(X.shape[1])
    del X

    # 2. Cells: chunk sweep at --max-iter, then the ladder at --ladder-chunk.
    cells: list[tuple[int | str, int]] = [(c, args.max_iter) for c in chunks]
    cells += [(args.ladder_chunk, it) for it in ladder if it != args.max_iter]

    base_env = {
        "JAX_PLATFORMS": "cuda" if args.device == "gpu" else "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "AMICA_DISABLE_EARLYSTOP": "1",      # iteration-matched: every cell runs its full budget
        "AMICA_NVML_CROSSCHECK": "1" if args.device == "gpu" else "0",
    }
    os.environ["SWEEP_JAX_PLATFORMS"] = base_env["JAX_PLATFORMS"]
    identity = _venv_identity(py)
    print(f"[sweep] runner venv: {py} -> {identity}")
    if "error" in identity:
        sys.exit("FATAL: cannot import jamica in the runner venv")

    manifest = {
        "_run": ip._orchestrator_run_block(),
        "sweep": {
            "dataset": args.dataset, "subject": args.subject, "subject_tag": subject_tag,
            "device": args.device, "n_components": args.n_components, "n_samples": n_samples,
            "chunks": [str(c) for c in chunks], "max_iter": args.max_iter,
            "ladder_iters": ladder, "ladder_chunk": args.ladder_chunk, "reps": args.reps,
            "seed": args.seed, "n_mix": args.n_mix, "lrate": args.lrate,
            "earlystop_disabled": True, "jit_included_in_fit_time": True,
            "runner": str(runner.relative_to(REPO_ROOT)), "venv_python": str(py),
        },
        "runner_identity": identity,
        "input_meta": meta,
        "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                                  "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE")},
        "cells": [],
    }
    manifest_path = dev_dir / f"manifest_{subject_tag}.json"

    def save_manifest() -> None:
        manifest["updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    for chunk, it in cells:
        for rep in range(1, args.reps + 1):
            key = "amica_python_jax" if chunk == FULL else "amica_python_jax_chunked"
            cell_dir = dev_dir / f"c{chunk}_i{it}_r{rep}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            out_json = cell_dir / f"{key}_{subject_tag}_seed{args.seed}_result.json"
            entry = {"chunk": str(chunk), "max_iter": it, "rep": rep, "key": key, "result": str(out_json)}
            if args.skip_existing and out_json.exists():
                try:
                    prev = json.loads(out_json.read_text())
                    if "error" not in prev and prev.get("n_iter") == it:
                        print(f"[sweep] skip existing {out_json.name} (chunk={chunk}, iter={it})")
                        entry["status"] = "skipped_existing"
                        manifest["cells"].append(entry)
                        continue
                except Exception:
                    pass
            env_extra = dict(base_env)
            if chunk != FULL:
                env_extra["AMICA_CHUNK_SIZE"] = str(chunk)
            cfg = dict(max_iter=it, n_mix=args.n_mix, lrate=args.lrate, do_newton=True, seed=args.seed)
            print(f"[sweep] cell chunk={chunk} iter={it} rep={rep} -> {out_json.name}")
            if args.dry_run:
                entry["status"] = "dry_run"
                manifest["cells"].append(entry)
                continue
            if out_json.exists():
                out_json.unlink()
            t0 = time.perf_counter()
            result = ip.run_subprocess(py, runner, input_path, out_json, cfg, env_extra, timeout_s=timeout)
            wall = time.perf_counter() - t0
            entry["wall_s"] = wall
            if "error" in result:
                entry["status"] = "error"
                entry["error"] = result.get("error")
                print(f"[sweep]   FAILED after {wall:.0f}s: {result.get('error')} {result.get('stderr', '')[-300:]}")
            else:
                entry["status"] = "ok"
                entry.update({k: result.get(k) for k in ("fit_time_s", "n_iter", "ll_final", "peak_rss_gb",
                                                          "peak_vram_gb", "nvml_peak_vram_gb", "nvml_post_init_gb")})
                print(f"[sweep]   ok: fit {result.get('fit_time_s', float('nan')):.1f}s, n_iter {result.get('n_iter')}, "
                      f"ll {result.get('ll_final', float('nan')):.5f}, rss {result.get('peak_rss_gb')} GiB, "
                      f"nvml {result.get('nvml_peak_vram_gb')} GiB (wall {wall:.0f}s)")
            manifest["cells"].append(entry)
            save_manifest()

    save_manifest()
    n_ok = sum(1 for c in manifest["cells"] if c.get("status") in ("ok", "skipped_existing"))
    print(f"[sweep] DONE {subject_tag} {args.device}: {n_ok}/{len(manifest['cells'])} cells ok -> {manifest_path}")
    return 0 if n_ok == len(manifest["cells"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
