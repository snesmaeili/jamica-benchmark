#!/usr/bin/env python3
"""Rewrite the jamica rows of gen_report.py from a re-measured sweep.

gen_report.py keeps every plotted median as a Python literal (the raw/*.csv files
are cited, not read). When only jamica is re-measured (the other implementations'
rows are reused), the jamica entries of each table have to be replaced in place
without touching the neighbours. This script reads the aggregate CSVs written by
benchmark/cc_benchmark/sweep/aggregate_chunk_sweep.py (raw/<tag>_*.csv) and
rewrites, for impl "jamica" only:

    GPU, GPU_BAND, GPU_CONV, LADDER, LADDER_MEM, MEM_CTX, MEM_LIVE, MEMDECOMP,
    GEXT, GEXT_BAND, MEM_RESV, the J_* two-key constants, CPU_FIT, CPU_BAND,
    CPU_NSUB, CPU_RSS, CPU_FB, CPU_LAD, MAIN, COMMIT["jamica"]

and records every inserted value in raw/<tag>_jamica_rows.json. Run from this
directory; --dry-run prints the replacements without writing:

    python patch_gen_report_jamica.py --tag v030 --commit v0.3.0 [--dry-run]

CPU tables are only rewritten when the CPU CSVs exist (the GPU and CPU sweeps
land separately).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CORE = [1024, 4096, 16384, 65536, 262144]        # gen_report's XT axis (FULL == 262144)
NAMES = {1024: "1024", 4096: "4096", 16384: "16384", 65536: "65536", 262144: "FULL",
         524288: "C512", 1048576: "C1M", "fullbatch": "FB"}


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x, nd=None):
    v = float(x)
    return round(v, nd) if nd is not None else v


def chunk_key(s: str):
    return "fullbatch" if s == "fullbatch" else int(s)


def rows_for(rows, impl, budget=None):
    return {chunk_key(r["chunk"]): r for r in rows if r["impl"] == impl}


# ----------------------------------------------------------------- value builders
def gpu_values(tag: str) -> dict:
    i3000 = rows_for(read(RAW / f"{tag}_gpu_i3000_summary.csv"), "jamica")
    i3000_fb = rows_for(read(RAW / f"{tag}_gpu_i3000_summary.csv"), "jamica_fullbatch")
    if not i3000:
        return {}
    ext = {chunk_key(r["chunk"]): r for r in read(RAW / f"{tag}_gpu_ext_summary.csv") if r["impl"] == "jamica"}
    ext_fb = {chunk_key(r["chunk"]): r for r in read(RAW / f"{tag}_gpu_ext_summary.csv") if r["impl"] == "jamica_fullbatch"}
    dec = rows_for(read(RAW / f"{tag}_gpumem_decomp.csv"), "jamica")
    dec_fb = rows_for(read(RAW / f"{tag}_gpumem_decomp.csv"), "jamica_fullbatch")
    percell = read(RAW / f"{tag}_gpu_percell.csv")
    ladder = {}
    for it in (100, 250, 500, 1000, 2000):
        r = rows_for(read(RAW / f"{tag}_gpu_i{it}_summary.csv"), "jamica").get(65536)
        if r:
            ladder[it] = r
    ladder[3000] = i3000[65536]

    v = {}
    v["GPU"] = {c: (f(i3000[c]["t_subj_median"], 1), f(i3000[c]["mem_median"], 2)) for c in CORE}
    v["GPU_BAND"] = {c: (round(f(i3000[c]["t_subj_p25"])), round(f(i3000[c]["t_subj_p75"]))) for c in CORE}
    r = i3000[262144]
    v["GPU_CONV"] = (f(r["ll_median"], 4), int(r["n_iter_min"]), int(float(r["n_iter_median"])), int(r["n_iter_max"]), f(r["s_per_iter_median"], 4))
    v["LADDER"] = {it: (f(ladder[it]["t_subj_median"], 1), f(ladder[it]["ll_median"], 5)) for it in sorted(ladder)}
    v["LADDER_MEM"] = {it: f(ladder[it]["mem_median"], 2) for it in sorted(ladder)}
    v["MEM_CTX"] = {c: f(dec[c]["context_nvml_post_init_gb"], 2) for c in CORE}
    v["MEM_LIVE"] = {c: f(dec[c]["live_alloc_gb"], 2) for c in CORE}
    v["MEMDECOMP"] = {c: (f(dec[c]["context_nvml_post_init_gb"], 2), f(dec[c]["live_alloc_gb"], 2), f(dec[c]["nvml_total_gb"], 2)) for c in (65536, 262144)}
    gext = {}
    for c, r in ((524288, ext.get(524288)), (1048576, ext.get(1048576)), ("fullbatch", ext_fb.get("fullbatch"))):
        if r:
            gext[c] = (f(r["fit"], 1), f(r["nvml"], 2), f(r["alloc"], 2), f(r["reserved"], 2) if r["reserved"] else None, int(r["n"]))
    v["GEXT"] = gext
    v["GEXT_BAND"] = {c: (f(rr["fit_p25"], 1), f(rr["fit_p75"], 1)) for c, rr in ((524288, ext.get(524288)), (1048576, ext.get(1048576)), ("fullbatch", ext_fb.get("fullbatch"))) if rr}
    v["MEM_RESV"] = {c: f(ext[c]["reserved"], 2) for c in (262144, 524288, 1048576) if c in ext and ext[c]["reserved"]}
    # two-key constants (chunked path at 262144 vs full-batch key)
    nv_ch = [f(r["nvml_gb"]) for r in percell if r["impl"] == "jamica" and r["chunk"] == "262144" and r["max_iter"] == "3000" and r["nvml_gb"]]
    nv_fb = [f(r["nvml_gb"]) for r in percell if r["impl"] == "jamica_fullbatch" and r["max_iter"] == "3000" and r["nvml_gb"]]
    v["J"] = {
        "J_CHUNKED_GPU_NVML": f(i3000[262144]["mem_median"], 2), "J_FULLBATCH_GPU_NVML": f(i3000_fb["fullbatch"]["mem_median"], 2),
        "J_CHUNKED_GPU_NVML_MAX": round(max(nv_ch), 2) if nv_ch else None, "J_FULLBATCH_GPU_NVML_MAX": round(max(nv_fb), 2) if nv_fb else None,
        "J_CHUNKED_ALLOC": f(dec[262144]["live_alloc_gb"], 2), "J_FULLBATCH_ALLOC": f(dec_fb["fullbatch"]["live_alloc_gb"], 2) if dec_fb else None,
        "J_CHUNKED_GPU_SPI": f(i3000[262144]["s_per_iter_median"], 4), "J_FULLBATCH_GPU_SPI": f(i3000_fb["fullbatch"]["s_per_iter_median"], 4),
    }
    return v


def cpu_values(tag: str) -> dict:
    i250 = rows_for(read(RAW / f"{tag}_cpu_i250_summary.csv"), "jamica")
    if not i250:
        return {}
    i250_fb = rows_for(read(RAW / f"{tag}_cpu_i250_summary.csv"), "jamica_fullbatch")
    ext = {chunk_key(r["chunk"]): r for r in read(RAW / f"{tag}_cpu_ext_summary.csv") if r["impl"] == "jamica"}
    ext_fb = {chunk_key(r["chunk"]): r for r in read(RAW / f"{tag}_cpu_ext_summary.csv") if r["impl"] == "jamica_fullbatch"}
    chunks = CORE + [524288, 1048576]
    v = {}
    v["CPU_FIT"] = {c: round(f(i250[c]["t_subj_median"])) for c in chunks if c in i250}
    v["CPU_BAND"] = {c: (round(f(i250[c]["t_subj_p25"])), round(f(i250[c]["t_subj_p75"]))) for c in chunks if c in i250}
    v["CPU_NSUB"] = {c: int(i250[c]["n_subjects"]) for c in chunks if c in i250}
    v["CPU_RSS"] = {c: f(i250[c]["mem_median"], 2) for c in chunks if c in i250}
    # the extension restricts the 1M point to recordings longer than the chunk, as published
    for c in (524288, 1048576):
        if c in ext:
            v["CPU_FIT"][c] = round(f(ext[c]["fit"])); v["CPU_BAND"][c] = (round(f(ext[c]["fit_p25"])), round(f(ext[c]["fit_p75"])))
            v["CPU_NSUB"][c] = int(ext[c]["n"]); v["CPU_RSS"][c] = f(ext[c]["rss"], 2)
    fb = ext_fb.get("fullbatch") or i250_fb.get("fullbatch")
    if fb:
        v["CPU_FB"] = (round(f(fb.get("fit") or fb.get("t_subj_median"))), f(fb.get("rss") or fb.get("mem_median"), 2), int(fb.get("n") or fb.get("n_subjects")))
    lad = {}
    for it in (50, 100, 500):
        r = rows_for(read(RAW / f"{tag}_cpu_i{it}_summary.csv"), "jamica").get(65536)
        if r:
            lad[it] = (f(r["t_subj_median"], 1), f(r["ll_median"], 4))
    lad[250] = (f(i250[65536]["t_subj_median"], 1), f(i250[65536]["ll_median"], 4))
    v["CPU_LAD"] = dict(sorted(lad.items()))
    v["J"] = {"J_CHUNKED_CPU_T": round(f(i250[4096]["t_subj_median"])), "J_FULLBATCH_CPU_T": v["CPU_FB"][0] if fb else None,
              "J_CHUNKED_CPU_RSS": f(i250[4096]["mem_median"], 1), "J_FULLBATCH_CPU_RSS": v["CPU_FB"][1] if fb else None}
    return v


# ----------------------------------------------------------------- literal formatting
def lit(x) -> str:
    if isinstance(x, tuple):
        return "(" + ",".join(lit(e) for e in x) + ")"
    if isinstance(x, dict):
        return "{" + ",".join(f"{NAMES.get(k, k)}:{lit(e)}" for k, e in x.items()) + "}"
    if x is None:
        return "None"
    if isinstance(x, float):
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if "." in s else s + ".0"
    return str(x)


def replace_jamica_value(line: str, new_literal: str) -> str:
    """Replace the value after the `"jamica":` key on this line (tuple or dict literal)."""
    k = line.index('"jamica":')
    i = k + len('"jamica":')
    while line[i] == " ":
        i += 1
    open_ch = line[i]
    close_ch = {"(": ")", "{": "}"}[open_ch]
    depth, j = 0, i
    while j < len(line):
        if line[j] == open_ch:
            depth += 1
        elif line[j] == close_ch:
            depth -= 1
            if depth == 0:
                break
        j += 1
    return line[:i] + new_literal + line[j + 1:]


def patch_dict(lines: list[str], name: str, new_literal: str, log: list[str]) -> None:
    start = next(k for k, l in enumerate(lines) if re.match(rf"^{name}\s*=\s*\{{", l))
    for k in range(start, len(lines)):
        if '"jamica":' in lines[k]:
            old = lines[k]
            lines[k] = replace_jamica_value(lines[k], new_literal)
            log.append(f"{name}: {old.strip()[:90]}  ->  {lines[k].strip()[:90]}")
            return
        if lines[k].startswith("}"):
            break
    raise SystemExit(f"{name}: no jamica entry found")


def patch_constants(lines: list[str], names_values: dict, log: list[str]) -> None:
    """Rewrite `A, B = x, y   # comment` style lines whose names are all in names_values."""
    for k, l in enumerate(lines):
        m = re.match(r"^([A-Z_0-9]+(?:\s*,\s*[A-Z_0-9]+)*)\s*=\s*([^#]*?)\s*(#.*)?$", l)
        if not m:
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        if not all(n in names_values for n in names):
            continue
        vals = ", ".join(lit(names_values[n]) for n in names)
        comment = ("   " + m.group(3)) if m.group(3) else ""
        new = f"{m.group(1)} = {vals}{comment}"
        log.append(f"{', '.join(names)}: {l.strip()[:80]}  ->  {new.strip()[:80]}")
        lines[k] = new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v030")
    ap.add_argument("--commit", default="v0.3.0", help='COMMIT["jamica"] label (tag or short SHA)')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpu = gpu_values(args.tag)
    cpu = cpu_values(args.tag)
    if not gpu and not cpu:
        raise SystemExit(f"no {args.tag}_* summaries under {RAW}")
    src = HERE / "gen_report.py"
    lines = src.read_text(encoding="utf-8").split("\n")
    log: list[str] = []

    if gpu:
        patch_dict(lines, "GPU", lit(gpu["GPU"]), log)
        patch_dict(lines, "GPU_BAND", lit(gpu["GPU_BAND"]), log)
        patch_dict(lines, "GPU_CONV", lit(gpu["GPU_CONV"]), log)
        patch_dict(lines, "LADDER", lit(gpu["LADDER"]), log)
        patch_dict(lines, "LADDER_MEM", lit(gpu["LADDER_MEM"]), log)
        patch_dict(lines, "MEM_CTX", lit(gpu["MEM_CTX"]), log)
        patch_dict(lines, "MEM_LIVE", lit(gpu["MEM_LIVE"]), log)
        patch_dict(lines, "MEMDECOMP", lit(gpu["MEMDECOMP"]), log)
        patch_dict(lines, "GEXT", lit(gpu["GEXT"]), log)
        patch_dict(lines, "GEXT_BAND", lit(gpu["GEXT_BAND"]), log)
        patch_dict(lines, "MEM_RESV", lit(gpu["MEM_RESV"]), log)
        patch_constants(lines, gpu["J"], log)
    if cpu:
        patch_dict(lines, "CPU_FIT", lit(cpu["CPU_FIT"]), log)
        patch_dict(lines, "CPU_BAND", lit(cpu["CPU_BAND"]), log)
        patch_dict(lines, "CPU_NSUB", lit(cpu["CPU_NSUB"]), log)
        patch_dict(lines, "CPU_RSS", lit(cpu["CPU_RSS"]), log)
        if "CPU_FB" in cpu:
            patch_dict(lines, "CPU_FB", lit(cpu["CPU_FB"]), log)
        patch_dict(lines, "CPU_LAD", lit(cpu["CPU_LAD"]), log)
        patch_constants(lines, cpu["J"], log)
        # the constants' inline comment described the earlier 1000-iteration CPU run
        for k, l in enumerate(lines):
            if l.startswith("J_CHUNKED_CPU_T, J_FULLBATCH_CPU_T") and "@1000" in l:
                lines[k] = l.replace("@1000", "@250 (whole-node Narval run)")
    # headline summary strings
    if gpu and cpu:
        gpu_best = min(v[0] for v in gpu["GPU"].values()); gpu_lo = min(v[1] for v in gpu["GPU"].values())
        gpu_hi = gpu["GEXT"]["fullbatch"][1] if "fullbatch" in gpu["GEXT"] else max(v[1] for v in gpu["GPU"].values())
        cpu_best = min(cpu["CPU_FIT"].values()); cpu_lo = min(cpu["CPU_RSS"].values()); cpu_hi = cpu["CPU_FB"][1] if "CPU_FB" in cpu else max(cpu["CPU_RSS"].values())
        main_t = (f"~{5*round(gpu_best/5):.0f} s", f"{gpu_lo:.0f}–{gpu_hi:.0f} GiB", f"~{round(cpu_best/60):.0f} min", f"{cpu_lo:.0f}–{cpu_hi:.0f} GiB")
        patch_dict(lines, "MAIN", "(" + ", ".join(f'"{s}"' for s in main_t) + ")", log)
    # build label
    for k, l in enumerate(lines):
        if l.startswith("COMMIT=") or l.startswith("COMMIT ="):
            new = re.sub(r'"jamica":"[^"]*"', f'"jamica":"{args.commit}"', l)
            log.append(f"COMMIT: {l.strip()[:60]}  ->  {new.strip()[:60]}")
            lines[k] = new

    print("\n".join(log))
    sidecar = {"tag": args.tag, "commit_label": args.commit, "gpu": gpu, "cpu": cpu}
    if args.dry_run:
        print(f"\n[dry-run] {len(log)} replacements; gen_report.py untouched")
        return 0
    src.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (RAW / f"{args.tag}_jamica_rows.json").write_text(json.dumps(sidecar, indent=1, default=str), encoding="utf-8", newline="\n")
    print(f"\nwrote gen_report.py ({len(log)} replacements) and raw/{args.tag}_jamica_rows.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
