#!/bin/bash
#SBATCH --job-name=py_sweep
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn
source /scratch/yorguin/amica-benchmark-repro/.venv_competitors/bin/activate
python - <<'PY'
import torch, time, numpy as np, pyamica
nc, ns = 64, 200000
Xn = np.random.default_rng(0).standard_normal((nc, ns)); Xn = Xn - Xn.mean(1, keepdims=True)
Xt = torch.from_numpy(Xn.T).to("cuda")
print("cuda:", torch.cuda.get_device_name(0), "| pyamica chunk_t sweep, 20 iters")
for ct in [None, 65536, 16384, 4096, 1024]:
    torch.cuda.reset_peak_memory_stats()
    try:
        m = pyamica.AMICA(n_components=nc, n_models=1, n_mix=3, max_iter=20, lrate=0.1,
                          rho0=1.5, do_sphere=False, do_newton=True, newt_start=50,
                          device="cuda", dtype=torch.float64, fix_init=True, chunk_t=ct)
        t0=time.perf_counter(); m.fit(Xt); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        vram=torch.cuda.max_memory_allocated()/1024**3
        print("  chunk_t=%-7s %6.1fs (%5.0f ms/it) VRAM=%5.2fGB"%(ct, dt, dt/20*1000, vram))
    except Exception as e:
        print("  chunk_t=%-7s ERR %s: %s"%(ct, type(e).__name__, str(e)[:60]))
PY
