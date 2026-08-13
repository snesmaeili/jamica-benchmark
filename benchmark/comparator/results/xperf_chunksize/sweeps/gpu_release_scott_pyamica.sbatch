#!/bin/bash
#SBATCH --job-name=sp_sweep
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn
source /scratch/yorguin/amica-benchmark-repro/.venv_competitors/bin/activate
python - <<'PY'
import torch, time, numpy as np
nc, ns = 64, 200000
Xn = np.random.default_rng(0).standard_normal((nc, ns)); Xn = Xn - Xn.mean(1, keepdims=True)
print("cuda:", torch.cuda.get_device_name(0), "| 20 iters, 64x200k")
sizes = [None, 65536, 16384, 4096, 1024]   # None = full batch (their default)

print("\n=== scott-huberty (amica.AMICA, batch_size) ===")
from amica import AMICA as ScottAMICA
for bs in sizes:
    torch.cuda.reset_peak_memory_stats()
    try:
        kw = {} if bs is None else {"batch_size": bs}
        m = ScottAMICA(n_components=nc, n_mixtures=3, device="cuda", n_models=1,
                       mean_center=False, whiten=None, max_iter=20, do_newton=True,
                       newt_start=50, random_state=0, verbose=0, **kw)
        t0=time.perf_counter(); m.fit(Xn.T); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        vram=torch.cuda.max_memory_allocated()/1024**3
        print("  batch_size=%-7s %6.1fs (%5.0f ms/it) VRAM=%4.1fGB"%(bs, dt, dt/20*1000, vram))
    except Exception as e: print("  batch_size=%-7s ERR %s: %s"%(bs, type(e).__name__, str(e)[:55]))

print("\n=== pyamica (pyamica.AMICA, chunk_t) ===")
import pyamica
for ct in sizes:
    torch.cuda.reset_peak_memory_stats()
    try:
        m = pyamica.AMICA(n_comps=nc, n_mix=3, max_iter=20, do_newton=True, newt_start=50,
                          rho0=1.5, do_sphere=False, device="cuda", dtype=torch.float64,
                          fix_init=True, chunk_t=ct)
        Xt = torch.from_numpy(Xn.T).to("cuda")
        t0=time.perf_counter(); m.fit(Xt); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        vram=torch.cuda.max_memory_allocated()/1024**3
        print("  chunk_t=%-7s %6.1fs (%5.0f ms/it) VRAM=%4.1fGB"%(ct, dt, dt/20*1000, vram))
    except Exception as e: print("  chunk_t=%-7s ERR %s: %s"%(ct, type(e).__name__, str(e)[:55]))
PY
