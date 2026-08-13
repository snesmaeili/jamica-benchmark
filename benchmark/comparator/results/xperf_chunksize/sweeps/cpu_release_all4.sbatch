#!/bin/bash
#SBATCH --job-name=cpu_sweep_rss
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --time=00:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/yorguin/%x-%j.out
#SBATCH --error=/scratch/yorguin/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
R=/scratch/yorguin/amica-benchmark-repro
export NC=64 NS=100000 IT=10
echo "=== CPU chunk sweep w/ peak RSS: 64x100000, 10 iters, 8 cores (fresh proc/config) ==="

module load StdEnv/2023 python/3.11 >/dev/null 2>&1

echo ">>> amica-python (chunk_size), CPU [JAX cpu]"
for CS in full auto 65536 16384 4096; do
  JAX_PLATFORMS=cpu CS=$CS $R/.venv_fir_gpu/bin/python - <<'PY'
import os,time,resource,numpy as np
cs_env=os.environ["CS"]; cs=None if cs_env=="full" else ("auto" if cs_env=="auto" else int(cs_env))
from amica import Amica,AmicaConfig
nc,ns,it=int(os.environ["NC"]),int(os.environ["NS"]),int(os.environ["IT"])
X=np.random.default_rng(0).standard_normal((nc,ns));X=X-X.mean(1,keepdims=True)
cfg=AmicaConfig(max_iter=it,num_mix_comps=3,lrate=0.1,do_newton=True,do_sphere=False,do_mean=False,chunk_size=cs)
m=Amica(cfg,random_state=0)
t=time.perf_counter();m.fit(X);d=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print("   chunk_size=%-8s %6.1fs (%5.0f ms/it) RSS=%5.2f GB"%(cs_env,d,d/it*1000,rss))
PY
done

source $R/.venv_competitors/bin/activate
echo ">>> scott-huberty (batch_size), CPU"
for BS in full 65536 16384 4096; do
  BS=$BS python - <<'PY'
import os,time,resource,numpy as np
bs_env=os.environ["BS"]; bs=None if bs_env=="full" else int(bs_env)
from amica import AMICA as S
nc,ns,it=int(os.environ["NC"]),int(os.environ["NS"]),int(os.environ["IT"])
X=np.random.default_rng(0).standard_normal((nc,ns));X=X-X.mean(1,keepdims=True)
kw={} if bs is None else {"batch_size":bs}
m=S(n_components=nc,n_mixtures=3,device="cpu",n_models=1,mean_center=False,whiten=None,max_iter=it,do_newton=True,newt_start=50,random_state=0,verbose=0,**kw)
t=time.perf_counter();m.fit(X.T);d=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print("   batch_size=%-8s %6.1fs (%5.0f ms/it) RSS=%5.2f GB"%(bs_env,d,d/it*1000,rss))
PY
done
echo ">>> pyamica (chunk_t), CPU"
for CT in full 65536 16384 4096; do
  CT=$CT python - <<'PY'
import os,time,resource,numpy as np,torch
ct_env=os.environ["CT"]; ct=None if ct_env=="full" else int(ct_env)
import pyamica
nc,ns,it=int(os.environ["NC"]),int(os.environ["NS"]),int(os.environ["IT"])
X=np.random.default_rng(0).standard_normal((nc,ns));X=X-X.mean(1,keepdims=True)
m=pyamica.AMICA(n_components=nc,n_models=1,n_mix=3,max_iter=it,lrate=0.1,rho0=1.5,do_sphere=False,do_newton=True,newt_start=50,device="cpu",dtype=torch.float64,fix_init=True,chunk_t=ct)
Xt=torch.from_numpy(X.T);t=time.perf_counter();m.fit(Xt);d=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print("   chunk_t=%-8s %6.1fs (%5.0f ms/it) RSS=%5.2f GB"%(ct_env,d,d/it*1000,rss))
PY
done
deactivate

module load python/3.12 >/dev/null 2>&1
source $R/.venv_pamica/bin/activate
echo ">>> pAMICA (block_size), CPU"
for BS in 512 4096 16384 65536 full; do
  BS=$BS python - <<'PY'
import os,time,resource,numpy as np,torch
bs_env=os.environ["BS"]
from pamica import AMICA
nc,ns,it=int(os.environ["NC"]),int(os.environ["NS"]),int(os.environ["IT"])
bs=ns if bs_env=="full" else int(bs_env)
X=np.random.default_rng(0).standard_normal((nc,ns));X=X-X.mean(1,keepdims=True)
m=AMICA(n_models=1,n_mix=3,device="cpu",verbose=False)
t=time.perf_counter();m.fit(X,max_iter=it,lrate=0.1,do_mean=False,do_sphere=True,do_newton=True,block_size=bs);d=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print("   block_size=%-8s %6.1fs (%5.0f ms/it) RSS=%5.2f GB"%(bs_env,d,d/it*1000,rss))
PY
done
echo CPU_SWEEP_RSS_DONE
