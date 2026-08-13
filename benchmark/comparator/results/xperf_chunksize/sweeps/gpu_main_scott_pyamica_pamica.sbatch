#!/bin/bash
#SBATCH --job-name=gpu_main_torch
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
R=/scratch/yorguin/amica-benchmark-repro
echo "=== GPU MAIN torch sweep: 64x200k, 20 iters, H100 (scott e15e1588, pyamica a8a4d7e, pAMICA 0c4da39) ==="
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn >/dev/null 2>&1
source $R/.venv_competitors_main/bin/activate
python - <<'PY'
import torch, time, numpy as np
nc, ns = 64, 200000
Xn = np.random.default_rng(0).standard_normal((nc, ns)); Xn = Xn - Xn.mean(1, keepdims=True)
print("cuda:", torch.cuda.get_device_name(0))
sizes=[None,65536,16384,4096,1024]
print(">>> scott-huberty MAIN (amica.AMICA, batch_size)")
from amica import AMICA as S
for bs in sizes:
    torch.cuda.reset_peak_memory_stats()
    try:
        kw={} if bs is None else {"batch_size":bs}
        m=S(n_components=nc,n_mixtures=3,device="cuda",n_models=1,mean_center=False,whiten=None,max_iter=20,do_newton=True,newt_start=50,random_state=0,verbose=0,**kw)
        t0=time.perf_counter(); m.fit(Xn.T); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        print("  batch_size=%-7s %6.1fs (%5.0f ms/it) VRAM=%4.2fGB"%("full" if bs is None else bs,dt,dt/20*1000,torch.cuda.max_memory_allocated()/1024**3))
    except Exception as e: print("  batch_size=%-7s ERR %s: %s"%(bs,type(e).__name__,str(e)[:55]))
print(">>> pyamica MAIN (pyamica.AMICA, chunk_t)")
import pyamica
for ct in sizes:
    torch.cuda.reset_peak_memory_stats()
    try:
        m=pyamica.AMICA(n_components=nc,n_models=1,n_mix=3,max_iter=20,lrate=0.1,rho0=1.5,do_sphere=False,do_newton=True,newt_start=50,device="cuda",dtype=torch.float64,fix_init=True,chunk_t=ct)
        Xt=torch.from_numpy(Xn.T).to("cuda")
        t0=time.perf_counter(); m.fit(Xt); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        print("  chunk_t=%-7s %6.1fs (%5.0f ms/it) VRAM=%4.2fGB"%("full" if ct is None else ct,dt,dt/20*1000,torch.cuda.max_memory_allocated()/1024**3))
    except Exception as e: print("  chunk_t=%-7s ERR %s: %s"%(ct,type(e).__name__,str(e)[:55]))
PY
deactivate
module load python/3.12 >/dev/null 2>&1
source $R/.venv_pamica_main/bin/activate
python - <<'PY'
import torch, time, numpy as np
from pamica import AMICA
nc, ns = 64, 200000
X = np.random.default_rng(0).standard_normal((nc, ns)); X = X - X.mean(1, keepdims=True)
print(">>> pAMICA MAIN (block_size)")
for bs in [512,4096,16384,65536,200000]:
    torch.cuda.reset_peak_memory_stats()
    m=AMICA(n_models=1,n_mix=3,device="cuda",verbose=False)
    try:
        t0=time.perf_counter(); m.fit(X,max_iter=20,lrate=0.1,do_mean=False,do_sphere=True,do_newton=True,block_size=bs); torch.cuda.synchronize(); dt=time.perf_counter()-t0
        print("  block_size=%-7s %6.1fs (%5.0f ms/it) VRAM=%5.2fGB"%("full" if bs==200000 else bs,dt,dt/20*1000,torch.cuda.max_memory_allocated()/1024**3))
    except Exception as e: print("  block_size=%-7s ERR %s: %s"%(bs,type(e).__name__,str(e)[:60]))
PY
echo GPU_MAIN_TORCH_DONE
