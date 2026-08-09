"""Runtime as a function of iteration count, per implementation and environment.

Each point is a complete fit run to a given iteration cap and timed end to end.
No implementation reports per-iteration times, and instrumenting four
third-party training loops would make every line a different measurement, so the
curve is built the one way that is uniform across all of them.

Two things the plot is careful about:

* The x value is the number of iterations *actually run*, not the cap. AMICA can
  stop early, and a fit that converged at 430 of a requested 700 belongs at 430.
  Plotting it at the cap would bend the curve downward for the best-converging
  implementation, which is exactly backwards.
* The slope in the legend is a least-squares fit over all the points, not a
  difference between the two extreme ones. A two-point slope inherits the full
  error of both endpoints, which is how a negative per-iteration cost ends up in
  a results table.

Usage::

    python plot_iter_curve.py --panel "Local CPU=results/comparator/itercurve_local_cpu_v2"
    python plot_iter_curve.py \\
        --panel "Cluster CPU=results/itercurve/cpu" \\
        --panel "Cluster GPU=results/itercurve/gpu" \\
        --out results/figures/fig_iter_curve_cluster.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Okabe-Ito, matching the rest of the figure set (make_main_figures.py).
BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
MAGENTA = "#CC79A7"
GREY = "#7A7A7A"
INK = "#202124"

# Display names as the cross-implementation table already uses them, so a reader
# moving between table and figure does not have to re-learn the row names.
STYLE = {
    "amica_python_jax_chunked": ("AMICA-Python (JAX, this work)", BLUE, "o", 2.4),
    "amica_python_jax": ("AMICA-Python (JAX, full batch)", LIGHT_BLUE, "v", 1.3),
    "scott_huberty_torch": ("AMICA-Python (PyTorch)", GREEN, "s", 1.3),
    "pyamica_torch": ("pyamica (PyTorch)", ORANGE, "D", 1.3),
    "pamica_torch": ("pAMICA (PyTorch)", MAGENTA, "^", 1.3),
    "fortran_amica17": ("AMICA 1.7 (Fortran)", GREY, "P", 1.3),
}


def load_panel(root: str) -> dict[str, list[tuple[int, float]]]:
    """Collect (iterations actually run, seconds) per implementation."""
    pts: dict[str, list[tuple[int, float]]] = defaultdict(list)
    pattern = os.path.join(root, "**", "*_result.json")
    for path in glob.glob(pattern, recursive=True):
        if "warmup" in os.path.basename(os.path.dirname(path)):
            continue  # discarded GPU warm-up fits are not measurements
        try:
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            continue
        if "error" in r or "fit_time_s" not in r:
            continue
        n_iter = r.get("n_iter") or r.get("max_iter")
        if not n_iter:
            continue
        pts[r["implementation"]].append((int(n_iter), float(r["fit_time_s"])))
    return {k: sorted(v) for k, v in pts.items()}


def slope_s_per_iter(points: list[tuple[int, float]]) -> float:
    """Least squares over every point, not a two-point difference."""
    if len(points) < 2:
        return float("nan")
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def canary_drift(path: str) -> str | None:
    """Summarise the between-block drift the canary recorded, if it was run.

    The first reading is dropped. Each canary is a fresh process, so the first
    one also pays a cold page cache on the JAX import and reads roughly twice
    the rest -- counting it would report a drift the campaign never saw. What is
    wanted is the spread across blocks once the machine is warm.
    """
    if not path or not os.path.exists(path):
        return None
    vals = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                vals.append(json.loads(line)["canary_s"])
            except Exception:
                pass
    if len(vals) < 3:
        return None
    warm = vals[1:]
    lo, hi = min(warm), max(warm)
    return (f"machine drift across blocks: {(hi / lo - 1) * 100:.0f}% "
            f"(canary {lo:.2f}-{hi:.2f} s, cold first reading {vals[0]:.2f} s excluded)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", action="append", required=True,
                    help="LABEL=ROOT, repeatable; one subplot per panel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--canary", default=None, help="canary jsonl to annotate drift")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for spec in args.panel:
        label, _, root = spec.partition("=")
        panels.append((label.strip(), root.strip(), load_panel(root.strip())))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4),
                             squeeze=False)
    for ax, (label, root, data) in zip(axes[0], panels):
        if not data:
            ax.text(0.5, 0.5, f"no results under\n{root}", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            ax.set_title(label, color=INK)
            continue
        order = sorted(data, key=lambda k: -slope_s_per_iter(data[k]))
        for impl in order:
            pts = data[impl]
            name, colour, marker, lw = STYLE.get(impl, (impl, GREY, ".", 1.2))
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            s = slope_s_per_iter(pts)
            ax.plot(x, y, marker=marker, color=colour, linewidth=lw,
                    markersize=5, label=f"{name}  ({s * 1000:.0f} ms/iter)")
        ax.set_xlabel("iterations")
        ax.set_ylabel("fit time (s)")
        ax.set_title(label, color=INK)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    drift = canary_drift(args.canary) if args.canary else None
    if drift:
        # A curve measured over hours is only meaningful if the machine held
        # steady, so say what it did rather than leaving the reader to assume.
        fig.text(0.005, 0.005, drift, fontsize=7, color=GREY)

    fig.tight_layout()
    out = args.out or "iter_curve.pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    png = os.path.splitext(out)[0] + ".png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}\nwrote {png}")
    if drift:
        print(drift)
    for label, _, data in panels:
        for impl, pts in sorted(data.items(), key=lambda kv: slope_s_per_iter(kv[1])):
            xs = ", ".join(str(p[0]) for p in pts)
            print(f"  {label:12} {impl:26} {slope_s_per_iter(pts) * 1000:8.1f} ms/iter  "
                  f"[{xs}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
