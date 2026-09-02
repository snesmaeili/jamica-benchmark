"""Cross-implementation cost table: CPU and GPU, matched fixture, 1000 iterations.

Rebuilt against the iteration-curve campaigns, which changed what this table can
honestly say. The previous version rested on two runs per implementation (100 and
600 iterations) and had to apologise for them at length in its own caption. Four
measured points per row, from one campaign per device, removes the apology:

  * **The headline is 1000 iterations, not 100.** AMICA needs on the order of a
    thousand iterations to converge, so a 100-iteration comparison measures
    start-up and compilation as much as it measures the algorithm -- and it was
    that regime, not the algorithm, that made the JAX implementation look slowest
    on GPU in the previous table.

  * **One estimator for every row.** Per-iteration cost is the slope of a
    least-squares fit through all four points. The old two-point form
    $(T_{600}-T_{100})/500$ inherits the full error of both endpoints, and when a
    600-iteration run finished faster than its own 100-iteration run it returned
    a *negative* cost, which forced a second estimator ($T_{600}/600$) for the
    rows where it broke. Two estimators in one column is not a column.

  * **Agreement is measured on converged fits.** The correlation columns compare
    the 1000-iteration decompositions, not the 100-iteration ones.

Run from figures/src/:  python make_tab_cross_implementation.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from _paths import DATA_ROOT as _WS, out

_HERE = Path(__file__).resolve().parent
DEST = out("tab_cross_implementation.tex")

# One directory per device, each holding iter<N>/ subdirectories.
CPU_ROOT = _WS / "results/comparator/cluster/cpu/itercurve_cpu"
GPU_ROOT = _WS / "results/comparator/cluster/gpu/itercurve_gpu"

HEADLINE_ITER = 1000

DISPLAY = {
    "amica_python_jax": r"\texttt{jamica} (JAX, full batch)",
    "amica_python_jax_chunked": r"\texttt{jamica} (JAX, chunked)",
    "scott_huberty_torch": r"AMICA-Python (PyTorch)",
    "pyamica_torch": r"\texttt{pyamica} (PyTorch)",
    "pamica_torch": r"\texttt{pAMICA} (PyTorch)",
    "fortran_amica17": r"Fortran AMICA~1.7",
}
REFERENCE = "amica_python_jax"  # agreement is quoted against the full-batch CPU run


def load_campaign(root: Path) -> dict[str, dict[int, dict]]:
    """{implementation: {iterations actually run: result record}}.

    Keyed by iterations *run* rather than requested: a fit that converged early
    belongs at the count it reached, and filing it under the requested cap would
    corrupt the slope for whichever implementation converges best.
    """
    campaign: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("iter*/*_result.json")):
        if "warmup" in path.parent.name:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if "error" in rec or "fit_time_s" not in rec:
            continue
        n_iter = int(rec.get("n_iter") or rec.get("max_iter"))
        campaign.setdefault(rec["implementation"], {})[n_iter] = rec
    return campaign


def per_iteration_ms(points: dict[int, dict]) -> tuple[float, int]:
    """Steady-state cost per iteration, from a least-squares fit over all points.

    The intercept absorbs the fixed start-up cost -- import, allocation,
    first-touch, compilation -- which is exactly what should not be charged to a
    per-iteration figure. Returns the slope and how many points produced it, so
    a thin row can be marked rather than quietly presented as equal evidence.
    """
    if len(points) < 2:
        return float("nan"), len(points)
    x = np.array(sorted(points), dtype=float)
    y = np.array([points[int(i)]["fit_time_s"] for i in x], dtype=float)
    return float(np.polyfit(x, y, 1)[0] * 1000), len(points)


def _matched_unmixing_summary(reference, candidate) -> tuple[float, float]:
    """Median and minimum unsigned row correlation after Hungarian matching.

    Same definition as the fixed-workload audit in make_main_figures.py; kept
    identical on purpose so the two cannot report different numbers for the
    same quantity.
    """
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    correlation = np.abs(reference @ candidate.T)
    row, col = linear_sum_assignment(-correlation)
    matched = correlation[row, col]
    return float(np.median(matched)), float(np.min(matched))


def agreement_at(campaign: dict[str, dict[int, dict]], n_iter: int) -> dict[str, dict]:
    """Agreement against the reference implementation's fit at the same budget.

    Compared at the converged budget rather than at 100 iterations: two
    implementations that will agree once converged can differ appreciably early,
    so an agreement number from a short fit describes the transient, not the
    decomposition.
    """
    ref = campaign.get(REFERENCE, {}).get(n_iter)
    if ref is None or "W" not in ref:
        return {}
    result: dict[str, dict] = {}
    for impl, points in campaign.items():
        rec = points.get(n_iter)
        if rec is None or "W" not in rec:
            continue
        _, worst = _matched_unmixing_summary(ref["W"], rec["W"])
        result[impl] = {
            "abs_ll_difference": abs(float(rec["ll_final"]) - float(ref["ll_final"])),
            "worst_matched_row_correlation": worst,
        }
    return result


def sci(value: float, digits: int = 1) -> str:
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}{{\times}}10^{{{int(exponent)}}}$"


def fixture_of(campaign: dict[str, dict[int, dict]]) -> tuple[int, int]:
    for points in campaign.values():
        for rec in points.values():
            return int(rec["n_components"]), int(rec["n_samples"])
    return 0, 0


def build_rows(campaign, agree, vram: bool, skipped: list[str]) -> list[str]:
    """Rows ordered by per-iteration cost, this package bolded wherever it lands."""
    built = []
    for impl, points in campaign.items():
        if impl not in DISPLAY:
            continue
        head = points.get(HEADLINE_ITER)
        per, n_points = per_iteration_ms(points)
        if head is None:
            skipped.append(f"{impl}@{HEADLINE_ITER}")
            time_cell, sort_key = "--", float("inf")
        else:
            time_cell, sort_key = f"${head['fit_time_s']:.1f}$", per
        if np.isnan(per):
            per_cell, sort_key = "--", float("inf")
        else:
            per_cell = f"${per:.0f}$" + (r"$^{*}$" if n_points < 4 else "")

        record = head or points[max(points)]
        rss = f"${float(record['peak_rss_gb']):.2f}$"
        if vram:
            v = record.get("peak_vram_gb")
            vram_cell = f"${float(v):.2f}$" if v else "--"
        else:
            vram_cell = "--"

        # The label only. Bolding a row's numbers would put a thumb on the
        # comparison the table exists to make.
        label = DISPLAY[impl]
        if impl.startswith("amica_python_"):
            label = rf"\textbf{{{label}}}"

        a = agree.get(impl)
        agree_cells = ("-- & --" if a is None else
                       f"{sci(a['abs_ll_difference'])} & "
                       f"${a['worst_matched_row_correlation']:.4f}$")
        built.append((sort_key,
                      f"{label} & {time_cell} & {per_cell} & {rss} & {vram_cell} & "
                      f"{agree_cells} \\\\"))
    built.sort(key=lambda b: b[0])
    return [line for _, line in built]


def main() -> None:
    cpu = load_campaign(CPU_ROOT)
    gpu = load_campaign(GPU_ROOT)
    if not cpu:
        raise SystemExit(f"no CPU campaign under {CPU_ROOT}")
    cpu_c, cpu_t = fixture_of(cpu)
    agree_cpu = agreement_at(cpu, HEADLINE_ITER)
    agree_gpu = agreement_at(gpu, HEADLINE_ITER) if gpu else {}
    # GPU rows are compared against the CPU reference, as before, so the two
    # blocks are commensurable.
    ref_cpu = cpu.get(REFERENCE, {}).get(HEADLINE_ITER)
    if ref_cpu is not None and gpu:
        agree_gpu = {}
        for impl, points in gpu.items():
            rec = points.get(HEADLINE_ITER)
            if rec is None or "W" not in rec:
                continue
            _, worst = _matched_unmixing_summary(ref_cpu["W"], rec["W"])
            agree_gpu[impl] = {
                "abs_ll_difference": abs(float(rec["ll_final"]) - float(ref_cpu["ll_final"])),
                "worst_matched_row_correlation": worst,
            }

    skipped: list[str] = []
    o: list[str] = []
    add = o.append
    add(r"% Generated by figures/src/make_tab_cross_implementation.py -- do not hand-edit.")
    add(r"\begin{table}[htbp]")
    add(r"\centering")
    add(r"\caption{")
    add(r"\textbf{Cross-implementation cost and agreement at a 1{,}000-iteration")
    add(r"budget.}")
    add(rf"Every implementation fitted the same PCA-projected Table tennis sub-01")
    add(rf"array (${cpu_c}\times{cpu_t:,}$".replace(",", "{,}") + r") to 100, 400, 700 and")
    add(r"1000 iterations. Fit time is quoted at 1000 iterations because AMICA")
    add(r"requires on the order of a thousand iterations to converge, so a")
    add(r"shorter budget measures start-up and compilation as much as it")
    add(r"measures the algorithm. Per-iteration cost is the slope of a")
    add(r"least-squares fit through all four points, the same estimator in every")
    add(r"row, with the intercept absorbing fixed start-up cost.")
    add(r"CPU rows ran on eight cores of a dual-socket AMD EPYC~9655 node with")
    add(r"thread counts bound to the allocation; GPU rows on one H100. All rows")
    add(r"of a given device come from a single campaign on one machine, and")
    add(r"component count, sample count, iteration budget and hardware are")
    add(r"identical within a block, so no row is advantaged by problem size.")
    add(r"Within each device, rows are ordered by per-iteration cost and")
    add(r"\texttt{jamica} is set in bold wherever that ordering places it.")
    add(r"The two right-hand columns compare each decomposition with the")
    add(r"\texttt{jamica} JAX-CPU fit at the same budget, Hungarian-matched and")
    add(r"sign-aligned.")
    add(r"\textbf{These are single runs, not repeated measurements}, and the")
    add(r"timing boundaries are not identical: the Python rows time the")
    add(r"model-fit call including first-use compilation, whereas the Fortran row")
    add(r"times the external executable including initialisation and output")
    add(r"writing but excluding input serialisation. Fixed start-up cost falls in")
    add(r"the intercept rather than the per-iteration column, so it affects the")
    add(r"fit-time column only.")
    add(r"\texttt{pAMICA} is run with its own algorithm constants and the shared")
    add(r"protocol (iteration budget, mixture count, learning rate, Newton")
    add(r"enabled). Each implementation ran at its own block-size setting, which")
    add(r"is a first-order determinant of cost rather than a detail: a separate")
    add(r"sweep archived with the benchmark code moves \texttt{pAMICA}'s fit time")
    add(r"by more than an order of magnitude across the range of block sizes these")
    add(r"packages expose, so no row here is that implementation's best achievable")
    add(r"cost.")
    add(r"The peak-memory columns are each framework's own allocator counter---peak")
    add(r"bytes in use for JAX, maximum bytes allocated for PyTorch---which are not")
    add(r"defined identically across frameworks and omit the CUDA context and the")
    add(r"pool the driver holds; whole-device measurement on the same fits puts that")
    add(r"understatement between $1.05$ and $3.34$ times, so these columns should be")
    add(r"read within a framework rather than across rows.")
    add(r"These measurements")
    add(r"predate the Anderson-accelerated variant of AMICA-Python, so this is a")
    add(r"descriptive comparison of one configuration rather than a durable")
    add(r"ranking of implementations.")
    add(r"}")
    add(r"\label{tab:cross-implementation}")
    add(r"\footnotesize")
    add(r"\setlength{\tabcolsep}{3.5pt}")
    add(r"\begin{tabular}{lcccccc}")
    add(r"\toprule")
    add(r"Implementation & \shortstack{Fit time\\(s, 1000 it.)} & "
        r"\shortstack{Per iteration\\(ms)} & \shortstack{Peak host\\RSS (GiB)} & "
        r"\shortstack{Peak\\VRAM (GiB)} & "
        r"\shortstack{$|\Delta$ final $\ell|$\\vs JAX-CPU} & "
        r"\shortstack{Worst matched\\row $|r|$} \\")
    add(r"\midrule")
    add(r"\multicolumn{7}{l}{\emph{CPU, eight cores}}\\")
    add(r"\addlinespace[1pt]")
    for line in build_rows(cpu, agree_cpu, vram=False, skipped=skipped):
        add(line)

    if gpu:
        add(r"\midrule")
        add(r"\multicolumn{7}{l}{\emph{GPU, one H100}}\\")
        add(r"\addlinespace[1pt]")
        for line in build_rows(gpu, agree_gpu, vram=True, skipped=skipped):
            add(line)

    add(r"\bottomrule")
    add(r"\end{tabular}")
    thin = any("$^{*}$" in line for line in o)
    if thin:
        add(r"")
        add(r"\vspace{2pt}")
        add(r"{\footnotesize $^{*}$ fitted from fewer than four iteration budgets.}")
    add(r"\end{table}")

    DEST.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {DEST}")
    for label, campaign in (("CPU", cpu), ("GPU", gpu)):
        if not campaign:
            continue
        print(f"  --- {label}")
        for impl, points in sorted(campaign.items(),
                                   key=lambda kv: per_iteration_ms(kv[1])[0]):
            per, n = per_iteration_ms(points)
            head = points.get(HEADLINE_ITER)
            print(f"  {DISPLAY.get(impl, impl):34} "
                  f"T{HEADLINE_ITER}={head['fit_time_s']:8.1f}s  " if head else
                  f"  {DISPLAY.get(impl, impl):34} T{HEADLINE_ITER}=      --  ",
                  end="")
            print(f"per-iter={per:8.1f}ms ({n} pts)")
    if skipped:
        print("  measurements absent, cells omitted: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
