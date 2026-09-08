#!/usr/bin/env python3
"""Generate tab_correctness.tex (Supplementary Table S8) from the campaign evidence files.

Sources, all produced from raw arrays of the jamica 0.3.0 campaign (September 2026):

* ``fig2_reference_evidence.json`` and ``fig2_reference_density.csv`` beside this script,
  written by ``scripts/v030_build_fig2_evidence.py`` in the validation workspace from the
  Fortran-parity ``parity.json`` files, the cross-backend agreement tables, the test-job
  outputs (full-batch/chunked pair, MNE interoperability) and the seed-robustness table;
* ``results/v030/agg/nearstock_parity_v030.json`` (``scripts/v030_nearstock_parity.py``): the
  free-initialisation Fortran agreement row.

The density statistics (Pearson r, nRMSE/IQR per parameter) are recomputed here from the
density CSV with the same formulas as make_main_figures.py, so the table never depends on
the order in which the figure and the table are regenerated. A partial evidence file
(``complete: false``) is refused.

Usage:
    python make_tab_correctness.py            # print
    python make_tab_correctness.py --write    # write <OUT_ROOT>/tab_correctness.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _paths import data, out

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "fig2_reference_evidence.json"
DENSITY = HERE / "fig2_reference_density.csv"
NEARSTOCK_JSON = data("results", "v030", "agg", "nearstock_parity_v030.json")
OUT = out("tab_correctness.tex")


def sci(x: float, sig: int = 2) -> str:
    """Render a number as LaTeX, using scientific notation when it is small."""
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    if -3 < exp < 4:
        s = f"{x:.{max(0, sig - 1 - exp)}f}".rstrip("0").rstrip(".")
        return f"\\({s}\\)"
    mant = x / (10 ** exp)
    m = f"{mant:.{sig - 1}f}".rstrip("0").rstrip(".")
    return f"\\({m}\\times10^{{{exp}}}\\)" if m != "1" else f"\\(10^{{{exp}}}\\)"


def pct(x: float, sig: int = 2) -> str:
    if x == 0:
        return "\\(0\\%\\)"
    exp = math.floor(math.log10(abs(x)))
    if -3 < exp < 3:
        s = f"{x:.{max(0, sig - 1 - exp)}f}".rstrip("0").rstrip(".")
        return f"\\({s}\\%\\)"
    mant = x / (10 ** exp)
    m = f"{mant:.{sig - 1}f}".rstrip("0").rstrip(".")
    return f"\\({m}\\times10^{{{exp}}}\\%\\)"


def r(x: float, dp: int) -> str:
    return f"\\({x:.{dp}f}\\)"


def rmin(x: float) -> str:
    """A correlation close to one, with enough decimals to show its distance from one."""
    if x >= 1.0:
        return r(1.0, 4)
    dp = int(min(11, max(4, math.ceil(-math.log10(1.0 - x)) + 1)))
    return r(x, dp)


def density_stats(csv_path: Path) -> dict[str, dict[str, float]]:
    """Pearson r and nRMSE/IQR per density parameter (same formulas as make_main_figures)."""
    import pandas as pd
    density = pd.read_csv(csv_path)
    stats = {}
    for key in ("alpha", "mu", "beta", "rho"):
        x = density[f"{key}_fortran"].to_numpy(float)
        y = density[f"{key}_python"].to_numpy(float)
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        stats[key] = {
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "nrmse_over_iqr": float(np.sqrt(np.mean((y - x) ** 2)) / iqr),
        }
    return stats, len(density)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if not ev.get("complete", False):
        raise SystemExit(f"{EVIDENCE.name} is a partial build (complete=false); rebuild it with "
                         "scripts/v030_build_fig2_evidence.py once every campaign input exists")
    near = json.loads(NEARSTOCK_JSON.read_text(encoding="utf-8"))
    panel = {e["label"].replace("\n", " "): e for e in ev["panel_a"]}
    dens, n_terms = density_stats(DENSITY)
    back, chunk, mne = ev["backend"], ev["chunking"], ev["mne"]

    def p(label):
        for k, v in panel.items():
            if label in k:
                return v
        raise SystemExit(f"panel_a row not found: {label}")

    k1, k3n, k3g = p("K=1, Newton"), p("K=3, Newton"), p("natural gradient")
    real = p("ds004505 sub-01")
    fixture_iters = int(k3n["fixture"]["max_iter"])
    fixture_samples = int(k3n["fixture"]["n_samples"])

    dr = min(v["pearson_r"] for v in dens.values())
    nl = min(v["nrmse_over_iqr"] for v in dens.values())
    nh = max(v["nrmse_over_iqr"] for v in dens.values())
    bmed = list(back["median_worst_row_correlation"].values())
    mir = float(back["maximum_absolute_mir_difference_kbits_s"])

    rows = [
        ("\\multirow{6}{=}{Fortran AMICA~1.7 parity}\n& 6-channel Laplacian, \\(K=1\\), Newton, fixed initialization",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k1['relative_ll_difference_pct'])}; {rmin(k1['worst_row_correlation'])}",
         f"Matched sources; {fixture_samples:,}-sample fixture, {fixture_iters:,} iterations"),
        ("& 6-channel Laplacian, \\(K=3\\), Newton, fixed initialization",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k3n['relative_ll_difference_pct'], 3)}; {rmin(k3n['worst_row_correlation'])}",
         f"Shared sphering and mean; {fixture_iters:,} iterations"),
        ("& 6-channel Laplacian, \\(K=3\\), natural gradient, fixed initialization",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k3g['relative_ll_difference_pct'])}; {rmin(k3g['worst_row_correlation'])}",
         "Matched sources; single fixture"),
        ("& 6-channel, \\(K=3\\), density parameters",
         "Matched \\(r\\); nRMSE/IQR for \\(\\alpha,\\mu,\\beta,\\rho\\)",
         f"\\(r\\geq{dr:.8f}\\);\n{sci(nl)}--{sci(nh)}",
         f"{n_terms} aligned density terms"),
        ("& 6-channel Laplacian, \\(K=3\\), Newton, free initialization on both sides",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(near['rel_ll_delta_pct'])}; {rmin(near['W_matched_abs_r_min'])}",
         f"Converged solutions compared in sensor space; {int(near['config']['max_iter']):,} iterations"),
        ("& Table tennis sub-01, 64 PCs, 100 iterations",
         "Absolute \\(\\Delta LL\\) (relative);\n\\(\\bar r_W\\) (\\(r_W^{\\min}\\))",
         f"{sci(real['absolute_ll_difference'])}\n({pct(real['relative_ll_difference_pct'])});\n"
         f"{r(real['mean_row_correlation'], 5)} ({rmin(real['worst_row_correlation'])})",
         "One subject; single model"),
    ]

    body = "\n\n".join(f"{a}\n& {b}\n& {c}\n& {d} \\\\" for a, b, c, d in rows)

    body += f"""

\\midrule

Cross-backend agreement
& Table tennis, {back['n_subjects']} participants; JAX-GPU, JAX-CPU, NumPy-CPU
& Median \\(r_W^{{\\min}}\\);
\\(\\lvert\\Delta\\mathrm{{MIR}}\\rvert_{{\\max}}\\)
& \\({min(bmed):.6f}\\)--\\({max(bmed):.6f}\\);
\\({mir:.4f}\\)~kbits/s
& Benchmark-level consistency \\\\

\\midrule

Full-batch/chunked agreement
& MNE sample; {chunk['n_components']} PCs; {chunk['n_samples']:,} samples; {chunk['max_iter']} iterations; \\texttt{{float64}}; \\(B={chunk['chunk_size']:,}\\)
& Absolute final \\(\\Delta LL\\);
\\(\\lVert\\Delta W\\rVert_F/\\lVert W\\rVert_F\\)
& {sci(chunk['absolute_final_ll_difference'], 3)};
{sci(chunk['unmixing_frobenius_relative_error'], 3)}
& Separate agreement fixture \\\\

\\midrule

MNE-Python interoperability
& \\texttt{{Raw}}/\\texttt{{Epochs}}, sources, exclusion, reconstruction,
reordering, rank reduction, model access, and ICLabel
& Direct tests ({mne['tests_selected']} passed); CI on Python~{mne['python_ci_versions'].replace('-', '--')} and \
{'three' if mne['ci_operating_systems'] == 3 else mne['ci_operating_systems']} operating systems
& All tested workflows passed; reconstruction residual
{sci(mne['transform_inverse_frobenius_relative_residual'])}
& Not a universal conformance claim \\\\"""

    tex = f"""% GENERATED by scripts/paper/figures/make_tab_correctness.py -- do not edit by hand.
% Sources: fig2_reference_evidence.json and fig2_reference_density.csv (built from the
% jamica 0.3.0 campaign outputs by scripts/v030_build_fig2_evidence.py) and
% results/v030/agg/nearstock_parity_v030.json (scripts/v030_nearstock_parity.py).
% Supplementary numerical audit supporting main Figure~\\ref{{fig:reference-agreement}}.
\\begin{{table}}[htbp]
\\centering
\\caption{{
\\textbf{{Numerical agreement and tested software interoperability.}}
Results are reported for the listed fixtures and configurations only.
}}
\\label{{tab:parity}}

\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}

\\begin{{tabular}}{{
    >{{\\raggedright\\arraybackslash}}p{{1.85cm}}
    >{{\\raggedright\\arraybackslash}}p{{3.35cm}}
    >{{\\raggedright\\arraybackslash}}p{{2.85cm}}
    >{{\\raggedright\\arraybackslash}}p{{3.05cm}}
    >{{\\raggedright\\arraybackslash}}p{{2.55cm}}
}}
\\toprule
Boundary & Fixture / configuration & Metric & Result & Scope \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}

\\vspace{{3pt}}
\\begin{{minipage}}{{\\linewidth}}
\\scriptsize
\\textit{{Notes.}}
\\(\\Delta LL\\) denotes either a relative percentage or an absolute difference
in normalized log-likelihood, as indicated. The normalized objective is
expressed in nats per retained component per sample.
\\(r_W^{{\\min}}\\) is the lowest Hungarian-matched, sign-aligned correlation
between unmixing rows within a fit. Parity results apply to single-model
AMICA (\\(M=1\\)) and to the listed fixtures only.
\\end{{minipage}}
\\end{{table}}
"""
    if args.write:
        OUT.write_text(tex, encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print(tex)

    print("\nrow sources:")
    for lbl, e in panel.items():
        print(f"  {lbl:44s} <- {e.get('source', '?')[:70]}")
    print(f"  {'free-initialisation row':44s} <- {NEARSTOCK_JSON}")
    print(f"  {'cross-backend':44s} <- {back.get('source', '?')[:70]}")
    print(f"  {'full-batch/chunked':44s} <- {chunk.get('source', '?')[:70]}")
    print(f"  {'MNE interoperability':44s} <- {mne.get('source', '?')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
