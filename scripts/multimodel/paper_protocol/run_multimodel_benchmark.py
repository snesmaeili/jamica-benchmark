"""Multi-model AMICA H-sweep runner for the stationarity benchmark (cluster GPU).

Paper protocol of the multi-model section (Figure 7, Supplementary Tables S6-S7,
Supplementary Figure S4), ported from the archived `amica-mm` campaign scripts to
the released `jamica` package. The algorithm is imported from `jamica` (the
installed release; fir_env.sh pins it); only the data loading reuses the vendored
harness `amica_python.benchmark.runner`.

Fits AMICA with H = --num-models on one subject of a dataset and saves everything
the LOCAL metric driver (compute_multimodel_metrics.py) and the manuscript
producers need: per-model sensor-space unmixing/mixing, model weights gm, the
model-posterior time course p(h|t) (block-averaged), LL history, task events,
channel names, and the data-length bookkeeping (N, n_samples, H_max, kappa_eff,
flag_underpowered). Output name and fields are those the producers parse:
    mmbench_<dataset>_sub-<NN>_N<N>_M<H>[_tentwenty][_surrphase].npz

Uses a SMALL n_components (default 16) and the FULL recording (--duration-sec 0)
so the data-length rule (~25*H*N^2 samples) supports H up to 10.

Run inside an allocation that sourced benchmark/cc_benchmark/fir_env.sh:
  JAX_PLATFORMS=cuda python run_multimodel_benchmark.py \
      --dataset ds004505 --subject 1 --num-models 3 --n-components 16 \
      --duration-sec 0 --resample 250 --max-iter 2000 --output-dir $RESULTS
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

K_DATALEN = 25  # Hsu's empirical constant in ~k*H*N^2

# Canonical 19-channel 10-20 set (+ legacy aliases) for the matched-channel control:
# subsetting the high-density task recording to the same montage as the 19-ch resting
# cohort isolates the task effect from electrode density.
TENTWENTY = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
             "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
_TT_ALIASES = {"T7": ("T7", "T3"), "T8": ("T8", "T4"), "P7": ("P7", "T5"), "P8": ("P8", "T6")}

# BIDS events live with the recording; only the task dataset has trial structure.
_EVENTS_ROOT_ENV = {"ds004505": ("BIDS_ROOT_DS4505", "TableTennis")}


def _harness_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _load_events(dataset: str, subject_id: int, window_sec):
    """Read the BIDS events.tsv (onset, duration, trial_type) of the task dataset.

    Returns onsets/durations in seconds and trial_type labels, cropped to
    [0, window_sec] if a window was applied. Empty arrays when the dataset has no
    events file (the resting cohorts) or the BIDS root is not configured.
    """
    empty = (np.array([]), np.array([]), np.array([], dtype=object))
    spec = _EVENTS_ROOT_ENV.get(dataset)
    if spec is None:
        return empty
    env_name, task = spec
    bids = os.environ.get(env_name)
    if not bids:
        return empty
    f = Path(bids) / f"sub-{subject_id:02d}" / "eeg" / f"sub-{subject_id:02d}_task-{task}_events.tsv"
    if not f.exists():
        return empty
    onsets, durs, types = [], [], []
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_on = header.index("onset") if "onset" in header else 0
        i_du = header.index("duration") if "duration" in header else 1
        i_ty = header.index("trial_type") if "trial_type" in header else (2 if len(header) > 2 else 1)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_on, i_ty):
                continue
            try:
                on = float(parts[i_on])
            except ValueError:
                continue
            du = 0.0
            try:
                du = float(parts[i_du])
            except (ValueError, IndexError):
                pass
            ty = parts[i_ty] if i_ty < len(parts) else ""
            if window_sec is not None and on > window_sec:
                continue
            onsets.append(on)
            durs.append(du)
            types.append(ty)
    return np.asarray(onsets), np.asarray(durs), np.asarray(types, dtype=object)


def _pick_tentwenty(raw):
    """Restrict raw to the 19 ten-twenty channels (name-matched, case-insensitive)."""
    have = {c.upper(): c for c in raw.info["ch_names"]}
    keep = []
    for name in TENTWENTY:
        for alias in _TT_ALIASES.get(name, (name,)):
            if alias.upper() in have:
                keep.append(have[alias.upper()])
                break
    if len(keep) < 16:
        raise RuntimeError(f"channel-subset tentwenty matched only {len(keep)}/19: {keep}")
    raw.pick(keep)
    return raw


def _phase_surrogate(X, seed):
    """Multivariate phase-randomized STATIONARY surrogate of X (N,T).

    Randomizes Fourier phases with the SAME random phase per frequency across all
    components, preserving every component's power spectrum and the cross-component
    (stationary) covariance while destroying temporal non-stationarity. A genuinely
    non-stationary recording yields N_eff/dLL well above its surrogate; an artifact of
    fitting many models to any data would not.
    """
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    Xf = np.fft.rfft(X, axis=1)
    ph = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=Xf.shape[1]))
    ph[0] = 1.0
    if T % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(Xf * ph[None, :], n=T, axis=1).astype(np.float64)


def _preprocess(dataset, subject, n_components, duration_sec, resample, input_level, seed,
                channel_subset=None, surrogate=None):
    """load -> (channel subset) -> (crop+resample) -> filter -> PCA(N) -> var-normalize
    -> (optional stationary surrogate).

    Returns (X (N,T), sfreq, ch_names, pca_components, pca_stds). duration_sec<=0 = full.
    """
    from sklearn.decomposition import PCA
    from amica_python.benchmark import runner as amica_runner

    raw, meta = amica_runner.load_data(
        dataset, subject, input_level=input_level, return_metadata=True
    )
    if channel_subset == "tentwenty":
        _pick_tentwenty(raw)
    win = duration_sec if (duration_sec and duration_sec > 0) else None
    if win is not None or resample:
        amica_runner.apply_analysis_window(raw, duration_sec=win, resample_sfreq=resample)
    # per-site mains notch (50 Hz for the European cohorts, 60 Hz for ds004505)
    raw = amica_runner.preprocess(raw, line_freq=meta.get("line_freq", 60.0))
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.info["ch_names"])

    data = raw.get_data().astype(np.float64)
    n_ch = data.shape[0]
    N = min(n_components, n_ch)
    pca = PCA(n_components=N, whiten=False, random_state=seed)
    projected = pca.fit_transform(data.T).T
    stds = np.std(projected, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X = projected / stds
    if surrogate == "phase":
        # stationary null: same data spectrum + covariance, non-stationarity removed
        X = _phase_surrogate(X, seed)
    # also return the PCA basis so sensor-space topographies can be reconstructed
    # downstream: sensor_mixing = components_.T @ (amica_mixing * stds)
    return (X, sfreq, ch_names,
            pca.components_.astype(np.float32), stds.ravel().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num-models", type=int, required=True)
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--duration-sec", type=float, default=0.0, help="0 = full recording")
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--input-level", default="bids")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--post-downsample-hz", type=float, default=10.0)
    ap.add_argument("--skip-underpowered", action="store_true")
    ap.add_argument("--channel-subset", default=None, choices=[None, "tentwenty"],
                    help="restrict to the 19 ten-twenty channels (matched-channel task control)")
    ap.add_argument("--surrogate", default=None, choices=[None, "phase"],
                    help="fit a multivariate phase-randomized stationary surrogate (null control)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    H = args.num_models
    suffix = ""
    if args.channel_subset:
        suffix += f"_{args.channel_subset}"
    if args.surrogate:
        suffix += f"_surr{args.surrogate}"
    tag = f"mmbench_{args.dataset}_sub-{args.subject:02d}_N{args.n_components}_M{H}{suffix}"
    out_path = out_dir / f"{tag}.npz"

    print(f"[mmbench] preprocessing {args.dataset} sub-{args.subject:02d} "
          f"(N={args.n_components}, dur={args.duration_sec or 'full'}, resample={args.resample})...")
    X, sfreq, ch_names, pca_components, pca_stds = _preprocess(
        args.dataset, args.subject, args.n_components, args.duration_sec,
        args.resample, args.input_level, args.seed,
        channel_subset=args.channel_subset, surrogate=args.surrogate,
    )
    N, T = X.shape
    H_max = int(T // (K_DATALEN * N * N))
    kappa_eff = float(T) / float(N * N)
    flag = H > H_max
    print(f"[mmbench] X=({N},{T}) sfreq={sfreq}  H_max={H_max} kappa_eff={kappa_eff:.1f} "
          f"H={H} {'UNDERPOWERED' if flag else 'ok'}")

    if flag and args.skip_underpowered:
        np.savez_compressed(out_path, skipped_underpowered=True, num_models=H,
                            n_components=N, n_samples=T, H_max=H_max,
                            kappa_eff=kappa_eff, sfreq=sfreq, subject=args.subject,
                            dataset=args.dataset)
        print(f"[mmbench] skipped (H>{H_max}); wrote stub {out_path}")
        return

    try:
        import jax
        device = "gpu" if any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm")
                              for d in jax.devices()) else "cpu"
    except Exception:
        device = "cpu"

    import jamica
    from jamica import Amica, AmicaConfig
    print(f"[mmbench] jamica {jamica.__version__} from {jamica.__file__}")
    cfg = AmicaConfig(num_models=H, max_iter=args.max_iter, num_mix_comps=args.num_mix,
                      do_newton=True, do_sphere=True, do_mean=True)
    print(f"[mmbench] fitting AMICA num_models={H} on {device} (chunk_size={cfg.chunk_size!r}) ...")
    result = Amica(cfg, random_state=args.seed).fit(X)

    ll_history = np.asarray(result.log_likelihood, dtype=np.float64)
    gm = np.atleast_1d(np.asarray(result.gm_, dtype=np.float64))
    v = result.model_posteriors_
    post = np.ones((1, T), dtype=np.float32) if v is None else np.asarray(v, dtype=np.float32)

    # downsampled posterior for compact figures (block-mean to ~post_downsample_hz)
    step = max(1, int(round(sfreq / max(args.post_downsample_hz, 1e-6))))
    n_blocks = post.shape[1] // step
    post_ds = (post[:, : n_blocks * step].reshape(post.shape[0], n_blocks, step).mean(axis=2)
               if n_blocks > 0 else post)

    # per-model sensor-space matrices (broadcast handles H=1 2D case)
    Wsens = np.asarray(result.unmixing_matrix_sensor_)
    Msens = np.asarray(result.mixing_matrix_sensor_)
    if Wsens.ndim == 2:
        Wsens = Wsens[None]
        Msens = Msens[None]

    on, du, ty = _load_events(args.dataset, args.subject,
                              None if args.duration_sec <= 0 else args.duration_sec)

    np.savez_compressed(
        out_path,
        dataset=args.dataset, subject=args.subject, device=device,
        channel_subset=str(args.channel_subset), surrogate=str(args.surrogate),
        num_models=H, n_components=N, n_samples=T, sfreq=sfreq,
        H_max=H_max, kappa_eff=kappa_eff, flag_underpowered=flag,
        samples_per_model=float(T) / H, n_iter=int(result.n_iter),
        max_iter=args.max_iter,
        ll_history=ll_history, ll_final=float(ll_history[-1]) if ll_history.size else float("nan"),
        gm=gm, model_posteriors_ds=post_ds, post_downsample_step=step,
        unmixing_matrix_sensor=Wsens.astype(np.float32),
        mixing_matrix_sensor=Msens.astype(np.float32),
        mean=np.asarray(result.mean_, dtype=np.float32),
        ch_names=np.array(ch_names, dtype=object),
        pca_components=pca_components, pca_stds=pca_stds,
        event_onsets=on, event_durations=du, event_types=ty,
        # provenance of this fit (the manuscript reports one release only)
        jamica_version=str(jamica.__version__), jamica_file=str(jamica.__file__),
        harness_commit=str(_harness_commit()), random_state=int(args.seed),
        chunk_size=str(cfg.chunk_size), num_mix_comps=int(args.num_mix),
    )
    print(f"[mmbench] saved {out_path}  (ll_final={float(ll_history[-1]):.4f}, "
          f"gm={np.round(gm, 3)}, N_eff~{np.exp(-(gm*np.log(np.clip(gm,1e-12,None))).sum()):.2f})")


if __name__ == "__main__":
    main()
