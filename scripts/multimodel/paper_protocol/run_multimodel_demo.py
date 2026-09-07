"""Multi-model AMICA demo on one real-EEG recording (Figure 7 posterior panel, Figure S4).

Fits AMICA with H = --num-models on one ds004505 subject and saves the model-probability
time course p(h|t), the log-likelihood history, the model weights and the task events, so
the manuscript producer can show whether the active model tracks the task structure and
whether H >= 2 raises the log-likelihood over H = 1. This is the archived `amica-mm`
`run_multimodel_demo.py` protocol (64 PCs, first 600 s at 250 Hz, 2,000 iterations, three
mixture components, seed 0) ported to the released `jamica` package; only the data
loading reuses the vendored harness `amica_python.benchmark.runner`.

Run inside an allocation that sourced benchmark/cc_benchmark/fir_env.sh:
    JAX_PLATFORMS=cuda python run_multimodel_demo.py \
        --subject 4 --num-models 3 --n-iter 2000 --n-components 64 \
        --num-mix 3 --duration-sec 600 --resample 250 --output-dir $RESULTS

Output: <output-dir>/mm_demo_sub-NN_M{H}.npz with the keys the producer parses
(num_models, n_components, n_samples, sfreq, model_posteriors, gm, event_onsets,
event_types, ...) plus the provenance of the fit (jamica_version, jamica_file,
harness_commit, chunk_size, seed).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_multimodel_benchmark import _load_events  # noqa: E402  (same protocol, same BIDS events)


def _harness_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _preprocess(subject_id: int, n_components: int, duration_sec, resample, seed: int):
    """ds004505 load -> analysis window -> filter -> PCA(n_components) -> unit variance.

    Mirrors the single-model benchmark preprocessing. Returns (projected (n_comp, T), sfreq).
    """
    from sklearn.decomposition import PCA

    from amica_python.benchmark import runner as amica_runner

    raw, meta = amica_runner.load_data(
        "ds004505", subject_id, input_level="bids", return_metadata=True
    )
    if duration_sec or resample:
        amica_runner.apply_analysis_window(
            raw, duration_sec=duration_sec or None, resample_sfreq=resample or None
        )
    raw = amica_runner.preprocess(raw, line_freq=meta.get("line_freq", 60.0))
    sfreq = float(raw.info["sfreq"])

    data = raw.get_data().astype(np.float64)
    n_comp = min(n_components, data.shape[0])
    pca = PCA(n_components=n_comp, whiten=False, random_state=seed)
    projected = pca.fit_transform(data.T).T
    stds = np.std(projected, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return projected / stds, sfreq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num-models", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mm-demo] preprocessing ds004505 sub-{args.subject:02d} "
          f"(n_comp={args.n_components}, dur={args.duration_sec}s, resample={args.resample})...")
    X, sfreq = _preprocess(args.subject, args.n_components, args.duration_sec, args.resample, args.seed)
    n_comp, n_samples = X.shape
    print(f"[mm-demo] X={X.shape}, sfreq={sfreq}")

    # Report the device JAX actually used (honest labelling).
    try:
        import jax
        device = "gpu" if any(
            getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in jax.devices()
        ) else "cpu"
    except Exception:
        device = "cpu"

    import jamica
    from jamica import Amica, AmicaConfig

    print(f"[mm-demo] jamica {jamica.__version__} from {jamica.__file__}")
    cfg = AmicaConfig(
        num_models=args.num_models,
        max_iter=args.n_iter,
        num_mix_comps=args.num_mix,
        do_newton=True,
        do_sphere=True,
        do_mean=True,
    )
    print(f"[mm-demo] fitting AMICA num_models={args.num_models} on {device} (chunk_size={cfg.chunk_size!r}) ...")
    result = Amica(cfg, random_state=args.seed).fit(X)

    ll_history = np.asarray(result.log_likelihood, dtype=np.float64)
    gm = np.atleast_1d(np.asarray(result.gm_, dtype=np.float64))
    v = result.model_posteriors_
    post = np.ones((1, n_samples), dtype=np.float32) if v is None else np.asarray(v, dtype=np.float32)

    on, du, ty = _load_events("ds004505", args.subject, args.duration_sec or None)

    out_path = out_dir / f"mm_demo_sub-{args.subject:02d}_M{args.num_models}.npz"
    np.savez_compressed(
        out_path,
        num_models=args.num_models,
        n_components=n_comp,
        n_samples=n_samples,
        sfreq=sfreq,
        device=device,
        n_iter=int(result.n_iter),
        ll_history=ll_history,
        ll_final=float(ll_history[-1]) if ll_history.size else float("nan"),
        gm=gm,
        model_posteriors=post,
        event_onsets=on,
        event_durations=du,
        event_types=ty,
        subject=args.subject,
        # provenance of this fit (the manuscript reports one release only)
        jamica_version=str(jamica.__version__),
        jamica_file=str(jamica.__file__),
        harness_commit=str(_harness_commit()),
        chunk_size=str(cfg.chunk_size),
        seed=args.seed,
        duration_sec=float(args.duration_sec),
        resample_hz=float(args.resample),
        num_mix=args.num_mix,
    )
    print(f"[mm-demo] saved {out_path}  (ll_final={float(ll_history[-1]):.4f}, gm={np.round(gm, 3)})")


if __name__ == "__main__":
    main()
