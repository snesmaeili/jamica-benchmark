#!/usr/bin/env python3
"""Aggregate the jamica block-size sweep (chunk_sweep_cell.py output) into the raw
summary CSVs that benchmark/comparator/results/xperf_chunksize/gen_report.py is
documented against (raw/*_summary.csv), plus the per-cell table.

LOCAL analysis step: rsync <out-root> from the clusters first, then

    python aggregate_chunk_sweep.py --root results/v030/sweep --tag v030 \
        --out-dir benchmark/comparator/results/xperf_chunksize/raw

Outputs (device in {gpu, cpu}; iter = each budget found):
    <tag>_<device>_percell.csv
        device,impl,chunk,rep,subject,fit_time_s,s_per_iter,ll_final,n_iter,max_iter,
        n_samples,mem_gb,nvml_gb,alloc_gb,pool_gb,ctx_gb,rss_gb,clamped
    <tag>_<device>_i<iter>_summary.csv      (schema of raw/nostop_gpu3000_summary.csv)
        device,impl,chunk,n_cells,n_subjects,subjects,t_pooled_median,t_subj_median,
        t_subj_p25,t_subj_p75,mem_median,s_per_iter_median,ll_median,n_iter_min,
        n_iter_median,n_iter_max,n_samples_min,n_samples_max
    <tag>_gpumem_summary.csv    impl,chunk,source_iter,n,nvml_median,nvml_min,nvml_max,alloc_median
    <tag>_gpumem_decomp.csv     impl,chunk,n,context_nvml_post_init_gb,live_alloc_gb,reserved_gb,nvml_total_gb
    <tag>_<device>_ext_summary.csv   impl,chunk,label,n,fit,fit_p25,fit_p75,spi,ll,ctx,alloc,reserved,nvml,rss
        (large chunks >= 262144 and full batch, restricted to subjects whose recording is
         longer than the chunk, as the published extension was)

impl names: "jamica" (chunked key) and "jamica_fullbatch" (full-batch key); chunk
"fullbatch" for the latter. Memory is GiB (bytes / 1024**3): GPU mem = NVML
whole-GPU peak (framework-neutral headline), CPU mem = process peak RSS.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CELL_RE = re.compile(r"^c(?P<chunk>\d+|full)_i(?P<iter>\d+)_r(?P<rep>\d+)$")
RES_RE = re.compile(r"^(?P<key>amica_python_jax(?:_chunked)?)_(?P<subject>sub-\d+|mne_sample)_seed(?P<seed>\d+)_result\.json$")
GIB = 1024 ** 3
EXT_CHUNKS = (262144, 524288, 1048576)
LABEL = {262144: "262K", 524288: "512K", 1048576: "1M", "fullbatch": "full"}


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _pct(vals, q):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.percentile(v, q)) if v else None


def _fmt(x, nd=6):
    return "" if x is None else (f"{x:.{nd}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x))


def load_cells(root: Path) -> list[dict]:
    rows = []
    for device in ("gpu", "cpu"):
        ddir = root / device
        if not ddir.is_dir():
            continue
        for cell_dir in sorted(ddir.iterdir()):
            m = CELL_RE.match(cell_dir.name)
            if not m:
                continue
            for f in sorted(cell_dir.glob("*_result.json")):
                rm = RES_RE.match(f.name)
                if not rm:
                    continue
                try:
                    d = json.loads(f.read_text())
                except Exception as e:  # unreadable cell: report, do not silently drop
                    print(f"[agg] unreadable {f}: {e}")
                    continue
                if "error" in d:
                    print(f"[agg] error cell {f}: {d.get('error')}")
                    continue
                chunk = "fullbatch" if m["chunk"] == "full" else int(m["chunk"])
                impl = "jamica_fullbatch" if rm["key"] == "amica_python_jax" else "jamica"
                n_iter = int(d.get("n_iter") or 0)
                fit = _f(d.get("fit_time_s"))
                n_samples = int(d.get("n_samples") or 0)
                vs = d.get("vram_stats") or {}
                pool = _f(vs.get("peak_pool_bytes"))
                rows.append({
                    "device": device, "impl": impl, "chunk": chunk, "rep": int(m["rep"]),
                    "subject": rm["subject"], "fit_time_s": fit,
                    "s_per_iter": (fit / n_iter) if (fit is not None and n_iter) else None,
                    "ll_final": _f(d.get("ll_final")), "n_iter": n_iter, "max_iter": int(d.get("max_iter") or m["iter"]),
                    "n_samples": n_samples,
                    "nvml_gb": _f(d.get("nvml_peak_vram_gb")), "alloc_gb": _f(d.get("peak_vram_gb")),
                    "pool_gb": (pool / GIB) if pool is not None else None,
                    "ctx_gb": _f(d.get("nvml_post_init_gb")), "rss_gb": _f(d.get("peak_rss_gb")),
                    "clamped": int(isinstance(chunk, int) and chunk >= n_samples > 0),
                    "budget": int(m["iter"]), "file": str(f),
                })
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"[agg] wrote {path} ({len(rows)} rows)")


def mem_of(r: dict):
    if r["device"] == "gpu":
        return r["nvml_gb"] if r["nvml_gb"] is not None else r["alloc_gb"]
    return r["rss_gb"]


def summarize(rows: list[dict]) -> list[list]:
    """One row per (device, impl, chunk): per-subject medians over reps, then medians over subjects."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["device"], r["impl"], r["chunk"])].append(r)
    out = []
    for (device, impl, chunk), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]).zfill(12))):
        by_subj: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_subj[r["subject"]].append(r)
        subj_t = [_med([x["fit_time_s"] for x in v]) for v in by_subj.values()]
        out.append([
            device, impl, chunk, len(rs), len(by_subj), "|".join(sorted(by_subj)),
            _fmt(_med([r["fit_time_s"] for r in rs])), _fmt(_med(subj_t)), _fmt(_pct(subj_t, 25)), _fmt(_pct(subj_t, 75)),
            _fmt(_med([mem_of(r) for r in rs])), _fmt(_med([r["s_per_iter"] for r in rs])), _fmt(_med([r["ll_final"] for r in rs])),
            min(r["n_iter"] for r in rs), _fmt(_med([float(r["n_iter"]) for r in rs])), max(r["n_iter"] for r in rs),
            min(r["n_samples"] for r in rs), max(r["n_samples"] for r in rs),
        ])
    return out


SUMMARY_HEADER = ["device", "impl", "chunk", "n_cells", "n_subjects", "subjects", "t_pooled_median", "t_subj_median",
                  "t_subj_p25", "t_subj_p75", "mem_median", "s_per_iter_median", "ll_median", "n_iter_min",
                  "n_iter_median", "n_iter_max", "n_samples_min", "n_samples_max"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="sweep out-root (contains gpu/ and/or cpu/)")
    ap.add_argument("--tag", default="v030")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "comparator" / "results" / "xperf_chunksize" / "raw")
    args = ap.parse_args()

    rows = load_cells(args.root)
    if not rows:
        raise SystemExit(f"no result cells under {args.root}")
    print(f"[agg] {len(rows)} cells")

    for device in ("gpu", "cpu"):
        drows = [r for r in rows if r["device"] == device]
        if not drows:
            continue
        write_csv(args.out_dir / f"{args.tag}_{device}_percell.csv",
                  ["device", "impl", "chunk", "rep", "subject", "fit_time_s", "s_per_iter", "ll_final", "n_iter", "max_iter",
                   "n_samples", "mem_gb", "nvml_gb", "alloc_gb", "pool_gb", "ctx_gb", "rss_gb", "clamped"],
                  [[r["device"], r["impl"], r["chunk"], r["rep"], r["subject"], _fmt(r["fit_time_s"]), _fmt(r["s_per_iter"]),
                    _fmt(r["ll_final"]), r["n_iter"], r["max_iter"], r["n_samples"], _fmt(mem_of(r)), _fmt(r["nvml_gb"]),
                    _fmt(r["alloc_gb"]), _fmt(r["pool_gb"]), _fmt(r["ctx_gb"]), _fmt(r["rss_gb"]), r["clamped"]] for r in drows])
        budgets = sorted({r["budget"] for r in drows})
        for it in budgets:
            brows = [r for r in drows if r["budget"] == it]
            write_csv(args.out_dir / f"{args.tag}_{device}_i{it}_summary.csv", SUMMARY_HEADER, summarize(brows))
        # large-chunk / full-batch extension: subjects longer than the chunk only
        main_it = max(budgets, key=lambda b: sum(1 for r in drows if r["budget"] == b))
        ext = []
        for chunk in (*EXT_CHUNKS, "fullbatch"):
            for impl in ("jamica", "jamica_fullbatch"):
                sel = [r for r in drows if r["budget"] == main_it and r["impl"] == impl and r["chunk"] == chunk
                       and (chunk == "fullbatch" or r["n_samples"] > chunk)]
                if not sel:
                    continue
                fits = [r["fit_time_s"] for r in sel]
                ext.append([impl, chunk, LABEL[chunk], len(sel), _fmt(_med(fits), 3), _fmt(_pct(fits, 25), 3), _fmt(_pct(fits, 75), 3),
                            _fmt(_med([r["s_per_iter"] for r in sel]), 4), _fmt(_med([r["ll_final"] for r in sel]), 4),
                            _fmt(_med([r["ctx_gb"] for r in sel]), 3), _fmt(_med([r["alloc_gb"] for r in sel]), 3),
                            _fmt(_med([r["pool_gb"] for r in sel]), 3), _fmt(_med([r["nvml_gb"] for r in sel]), 3),
                            _fmt(_med([r["rss_gb"] for r in sel]), 3)])
        write_csv(args.out_dir / f"{args.tag}_{device}_ext_summary.csv",
                  ["impl", "chunk", "label", "n", "fit", "fit_p25", "fit_p75", "spi", "ll", "ctx", "alloc", "reserved", "nvml", "rss"], ext)
        if device == "gpu":
            mem, dec = [], []
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for r in drows:
                if r["budget"] == main_it:
                    groups[(r["impl"], r["chunk"])].append(r)
            for (impl, chunk), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]).zfill(12))):
                nv = [r["nvml_gb"] for r in rs if r["nvml_gb"] is not None]
                mem.append([impl, chunk, main_it, len(rs), _fmt(_med(nv)), _fmt(min(nv) if nv else None), _fmt(max(nv) if nv else None),
                            _fmt(_med([r["alloc_gb"] for r in rs]))])
                dec.append([impl, chunk, len(rs), _fmt(_med([r["ctx_gb"] for r in rs]), 3), _fmt(_med([r["alloc_gb"] for r in rs]), 3),
                            _fmt(_med([r["pool_gb"] for r in rs]), 3), _fmt(_med(nv), 3)])
            write_csv(args.out_dir / f"{args.tag}_gpumem_summary.csv",
                      ["impl", "chunk", "source_iter", "n", "nvml_median", "nvml_min", "nvml_max", "alloc_median"], mem)
            write_csv(args.out_dir / f"{args.tag}_gpumem_decomp.csv",
                      ["impl", "chunk", "n", "context_nvml_post_init_gb", "live_alloc_gb", "reserved_gb", "nvml_total_gb"], dec)

    # console digest
    print("\ndevice impl              chunk        n  t_subj_med   s/iter    ll_med    mem_med")
    for row in summarize(rows):
        print(f"{row[0]:<6} {row[1]:<17} {str(row[2]):<10} {row[4]:>3}  {row[7]:>10}  {row[11]:>8}  {row[12]:>8}  {row[10]:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
