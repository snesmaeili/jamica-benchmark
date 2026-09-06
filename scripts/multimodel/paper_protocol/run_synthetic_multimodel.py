"""Synthetic tier: stationary vs non-stationary multi-model AMICA (Table S7, Fig 7).

Paper protocol ported from the archived `amica-mm` campaign to the released
`jamica` package. Generates a STATIONARY single-ICA-mixture and a NON-STATIONARY
concatenation of K distinct mixtures (known switch times + ground-truth A_k / S_k),
fits AMICA for H = 1..Hmax on each, and computes per H:
  - LL(H), N_eff(H), switching/dwell  (stationarity signature; amica_python.benchmark.stationarity)
  - ground-truth model error (Amari-style), SIR (dB), symmetric-KL of source PDFs
    (per regime, matched to its dominant model; benchmark/mne_synthetic/score_ground_truth)

Expectation (Hsu et al. 2018): non-stationary -> error/SIR/KL improve up to H=K then
flatten, N_eff>1, switching present; stationary -> flat in H, N_eff~1, ~no switching.
Saves synthetic_summary.json (+ a contrast figure when matplotlib is available).

Published protocol (figdata/multimodel_synthetic_2000, seeds 0-9):
  python run_synthetic_multimodel.py --n-components 16 --tseg 25000 --n-regimes 3 \
      --max-h 10 --max-iter 2000 --seed 0 --out-dir <root>/synthetic/seed0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "benchmark" / "mne_synthetic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import score_ground_truth as sgt  # noqa: E402  (benchmark/mne_synthetic/score_ground_truth.py)
from amica_python.benchmark import stationarity as st  # noqa: E402  (harness metrics)

import jamica  # noqa: E402
from jamica import Amica, AmicaConfig  # noqa: E402


def _harness_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _gen_regime(n, T, rng):
    """One stationary ICA regime: distinct random mixing of Laplacian sources."""
    A = rng.standard_normal((n, n))
    S = rng.laplace(size=(n, T))
    return A, S


def gen_stationary(n, T, seed):
    rng = np.random.default_rng(seed)
    A, S = _gen_regime(n, T, rng)
    X = A @ S
    regimes = [{"A": A, "S": S, "sl": slice(0, T)}]
    return X, regimes


def gen_nonstationary(n, Tseg, K, seed):
    rng = np.random.default_rng(seed + 100)
    blocks, regimes = [], []
    for k in range(K):
        A, S = _gen_regime(n, Tseg, rng)
        blocks.append(A @ S)
        regimes.append({"A": A, "S": S, "sl": slice(k * Tseg, (k + 1) * Tseg)})
    X = np.concatenate(blocks, axis=1)
    return X, regimes


def _per_model_sources(result, X):
    """Recovered sources per model in sensor space: (H, n, T)."""
    mean_ = np.asarray(result.mean_)
    Wsens = np.asarray(result.unmixing_matrix_sensor_)  # (H,n,nch) for H>1 else (n,nch)
    if Wsens.ndim == 2:
        Wsens = Wsens[None]
    Xc = X - mean_[:, None]
    return np.stack([Wsens[h] @ Xc for h in range(Wsens.shape[0])])


def _score_fit(result, X, regimes, sfreq):
    """Ground-truth + stationarity metrics for one fitted H-model result."""
    gm = np.atleast_1d(np.asarray(result.gm_, dtype=float))
    H = gm.shape[0]
    ll = float(np.asarray(result.log_likelihood)[-1])

    # posteriors p(h|t): (H, T); single-model -> a row of ones
    v = result.model_posteriors_
    T = X.shape[1]
    v = np.ones((1, T)) if v is None else np.asarray(v, dtype=float)

    mix = np.asarray(result.mixing_matrix_sensor_)   # (H,nch,n) or (nch,n)
    if mix.ndim == 2:
        mix = mix[None]
    src = _per_model_sources(result, X)              # (H, n, T)

    # For each true regime, pick its dominant model (max mean posterior in-regime),
    # then score that model's recovery of the regime's sources/topographies.
    errs, sirs, kls = [], [], []
    for reg in regimes:
        sl = reg["sl"]
        h = 0 if H == 1 else int(v[:, sl].mean(axis=1).argmax())
        topo = sgt.hungarian_match_topographies(reg["A"], mix[h])
        col, sgn = topo["col_idx_in_hat"], topo["signs"]
        S_hat_reg = src[h][:, sl]
        errs.append(1.0 - float(np.mean(topo["r_topo_abs"])))   # model error = 1 - mean matched |r_topo|
        sirs.append(float(np.median(sgt.sir_db(reg["S"], S_hat_reg, col, sgn))))
        kls.append(float(np.nanmedian(sgt.symmetric_kl_pdf(reg["S"], S_hat_reg, col, sgn))))

    summ = st.stationarity_summary(gm, v, sfreq)
    return {
        "H": H, "ll": ll,
        "model_error": float(np.mean(errs)),        # 1 - mean matched |r_topo|
        "sir_db": float(np.mean(sirs)),
        "symmetric_kl": float(np.mean(kls)),
        "n_eff": summ["n_eff"],
        "switching_rate_hz": summ["switching_rate_hz"],
        "mean_dwell_s": summ["mean_dwell_s"],
        "posterior_entropy_mean": summ["posterior_entropy_mean"],
        "gm": gm.tolist(),
        "n_iter": int(result.n_iter),
    }


def run_dataset(name, X, regimes, max_h, max_iter, seed, sfreq):
    print(f"\n=== {name}: X={X.shape}, regimes={len(regimes)} ===")
    rows = []
    for H in range(1, max_h + 1):
        cfg = AmicaConfig(num_models=H, max_iter=max_iter, num_mix_comps=3,
                          do_newton=True, do_sphere=True, do_mean=True)
        res = Amica(cfg, random_state=seed).fit(X)
        row = _score_fit(res, X, regimes, sfreq)
        rows.append(row)
        print(f"  H={H}: LL={row['ll']:.4f}  err={row['model_error']:.3f}  "
              f"SIR={row['sir_db']:.1f}dB  KL={row['symmetric_kl']:.3f}  "
              f"N_eff={row['n_eff']:.2f}  switch={row['switching_rate_hz']:.3f}Hz  n_iter={row['n_iter']}")
    return rows


def _plot(stat_rows, ns_rows, out_dir, n_regimes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # the figure is a convenience; the JSON is the deliverable
        print(f"[synthetic] matplotlib unavailable ({exc}); skipping the contrast figure")
        return None
    Hs = [r["H"] for r in ns_rows]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    def series(rows, key):
        return [r[key] for r in rows]

    panels = [
        ("ll", "log-likelihood", "A. LL vs H"),
        ("model_error", "model error (1-|r_topo|)", "B. Model error vs H"),
        ("sir_db", "SIR (dB)", "C. SIR vs H"),
        ("symmetric_kl", "symmetric KL", "D. Source-PDF KL vs H"),
        ("n_eff", "N_eff (active models)", "E. N_eff vs H"),
        ("switching_rate_hz", "switching rate (Hz)", "F. Switching vs H"),
    ]
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        ax.plot(Hs, series(ns_rows, key), "o-", color="#e6550d", label="non-stationary")
        ax.plot(Hs, series(stat_rows, key), "s--", color="#3182bd", label="stationary")
        ax.axvline(n_regimes, color="#888", ls=":", lw=1, label=f"K={n_regimes}")
        ax.set_xlabel("Number of models, M"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Synthetic multi-model AMICA: stationary vs non-stationary "
                 f"(K={n_regimes} regimes)", fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, wspace=0.28, hspace=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "fig_synthetic_stationarity_contrast.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--tseg", type=int, default=25000)
    ap.add_argument("--n-regimes", type=int, default=3)
    ap.add_argument("--max-h", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--out-dir", type=Path, default=Path("results/multimodel_synthetic"))
    args = ap.parse_args()

    print(f"[synthetic] jamica {jamica.__version__} from {jamica.__file__}")
    n, K = args.n_components, args.n_regimes
    T_total = args.tseg * K
    X_stat, reg_stat = gen_stationary(n, T_total, args.seed)
    X_ns, reg_ns = gen_nonstationary(n, args.tseg, K, args.seed)

    stat_rows = run_dataset("STATIONARY (single mixture)", X_stat, reg_stat,
                            args.max_h, args.max_iter, args.seed, args.sfreq)
    ns_rows = run_dataset(f"NON-STATIONARY ({K} regimes)", X_ns, reg_ns,
                          args.max_h, args.max_iter, args.seed, args.sfreq)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args) | {"out_dir": str(args.out_dir)},
        "provenance": {
            "jamica_version": jamica.__version__, "jamica_file": jamica.__file__,
            "harness_commit": _harness_commit(),
            "chunk_size": str(AmicaConfig().chunk_size), "num_mix_comps": 3,
        },
        "stationary": stat_rows,
        "non_stationary": ns_rows,
        "delta_ll_stationary": st.delta_ll({r["H"]: r["ll"] for r in stat_rows}),
        "delta_ll_non_stationary": st.delta_ll({r["H"]: r["ll"] for r in ns_rows}),
    }
    (args.out_dir / "synthetic_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    p = _plot(stat_rows, ns_rows, args.out_dir, K)
    print(f"\n[synthetic] wrote {args.out_dir / 'synthetic_summary.json'}" + (f" and {p}" if p else ""))
    if K <= args.max_h:
        print(f"[synthetic] dLL(H=K) NS={summary['delta_ll_non_stationary'][K]['delta']:.3f} "
              f"vs stationary={summary['delta_ll_stationary'][K]['delta']:.3f}")


if __name__ == "__main__":
    main()
