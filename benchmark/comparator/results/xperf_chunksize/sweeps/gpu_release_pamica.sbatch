#!/bin/bash
#SBATCH --job-name=pamica_bs2
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.12 cuda/12.6 cudnn
source /scratch/yorguin/amica-benchmark-repro/.venv_pamica/bin/activate
python - <<'PY'
import torch, time, numpy as np
from pamica import AMICA
nc, ns = 64, 200000
X = np.random.default_rng(0).standard_normal((nc, ns))
X = X - X.mean(1, keepdims=True)
print("cuda:", torch.cuda.get_device_name(0), "| data (%d x %d)"%(nc,ns))
for bs in [512, 4096, 16384, 65536, 200000]:
    torch.cuda.reset_peak_memory_stats()
    m = AMICA(n_models=1, n_mix=3, device="cuda", verbose=False)
    t0=time.perf_counter()
    try:
        m.fit(X, max_iter=20, lrate=0.1, do_mean=False, do_sphere=True, do_newton=True, block_size=bs)
        torch.cuda.synchronize(); dt=time.perf_counter()-t0
        vram=torch.cuda.max_memory_allocated()/1024**3
        print("  block_size=%7d: %6.1fs (%6.0f ms/iter) VRAM=%5.2f GB"%(bs,dt,dt/20*1000,vram))
    except Exception as e:
        print("  block_size=%7d: ERROR %s: %s"%(bs,type(e).__name__,str(e)[:70]))
PY
