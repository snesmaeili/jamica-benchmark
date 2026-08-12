#!/usr/bin/env python3
"""Regenerate xperf_summary.json from the committed xperf_measures.csv.

The CSV is the ground-truth record (one row per device x subject x implementation,
fit-time + peak-memory). This script derives the aggregate stats and paired tests
so the numbers in xperf_summary.json / the report are auditable and reproducible
from the committed measures alone -- no cluster access or raw ICA output needed.

    python make_summary.py            # writes xperf_summary.json next to the CSV

Only depends on scipy + the standard library.
"""
from __future__ import annotations
import csv, json, statistics as st
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent
IMPLS = ["amica_python_jax", "amica_python_jax_chunked", "pyamica_torch",
         "scott_huberty_torch", "pamica_torch", "fortran_amica17"]


def load():
    data = {"cpu": {}, "gpu": {}}
    with open(HERE / "xperf_measures.csv") as fh:
        for r in csv.DictReader(fh):
            t = float(r["fit_time_s"]) if r["fit_time_s"] else None
            p = float(r["peak_mem_gb"]) if r["peak_mem_gb"] else None
            data[r["device"]].setdefault(r["subject"], {})[r["implementation"]] = {"t": t, "peak": p}
    return data


def agg(dev, imp):
    ts = [dev[s][imp]["t"] for s in dev if dev[s].get(imp) and dev[s][imp]["t"]]
    ps = [dev[s][imp]["peak"] for s in dev if dev[s].get(imp) and dev[s][imp].get("peak")]
    if not ts:
        return None
    return {"n": len(ts), "time_mean_s": round(st.mean(ts), 3), "time_std_s": round(st.pstdev(ts), 3),
            "time_min_s": round(min(ts), 3), "time_max_s": round(max(ts), 3),
            "peak_mem_gb_mean": round(st.mean(ps), 3) if ps else None}


def paired(dev, a, b):
    subs = [s for s in dev if dev[s].get(a, {}).get("t") and dev[s].get(b, {}).get("t")]
    av = [dev[s][a]["t"] for s in subs]
    bv = [dev[s][b]["t"] for s in subs]
    t, pt = stats.ttest_rel(bv, av)
    try:
        _, pw = stats.wilcoxon(bv, av)
    except Exception:
        pw = None
    return {"n": len(subs), "a_faster_in": sum(1 for x, y in zip(av, bv) if x < y),
            "a_mean": round(st.mean(av), 2), "b_mean": round(st.mean(bv), 2),
            "ttest_p": round(pt, 5), "wilcoxon_p": round(pw, 5) if pw is not None else None}


def main():
    d = load()
    cpu, gpu = d["cpu"], d["gpu"]
    summary = {
        "run": {"dataset": "ds004505", "n_subjects": 25, "n_components": 64, "max_iter": 100, "seeds": 1,
                "cpu": "def-kjerbi_cpu 8 cores", "gpu": "NVIDIA H100 80GB",
                "timing": "wall time of fit() incl JIT compile, cold-consistent (per-subject JAX_COMPILATION_CACHE_DIR)",
                "numerical_agreement_Wcorr": "0.997-1.000 vs Fortran reference"},
        "aggregate": {"cpu": {i: agg(cpu, i) for i in IMPLS if agg(cpu, i)},
                      "gpu": {i: agg(gpu, i) for i in IMPLS if agg(gpu, i)}},
        "paired_tests_gpu_vs_amica_chunked": {
            i: paired(gpu, "amica_python_jax_chunked", i)
            for i in ("scott_huberty_torch", "pyamica_torch", "pamica_torch")},
        "paired_tests_cpu_vs_amica_chunked": {
            i: paired(cpu, "amica_python_jax_chunked", i)
            for i in ("scott_huberty_torch", "pamica_torch")},
    }
    (HERE / "xperf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", HERE / "xperf_summary.json")


if __name__ == "__main__":
    main()
