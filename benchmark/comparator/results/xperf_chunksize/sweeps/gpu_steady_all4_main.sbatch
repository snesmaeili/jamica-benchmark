#!/bin/bash
#SBATCH --job-name=steady_all
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
R=/scratch/yorguin/amica-benchmark-repro
export MEAS=/scratch/yorguin/steady_meas.txt; : > $MEAS
echo "=== GPU steady-state via 2-point differencing (K=10,40; steady=(T40-T10)/30), all MAIN, 64x200k, H100 ==="

# ---------- amica (main via AMICA_SRC, JAX cuda) ----------
module load StdEnv/2023 python/3.11 scipy-stack/2026a cuda/12.6 cudnn >/dev/null 2>&1
[ -n "$CUDA_HOME" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
for CH in 1024 4096 16384 65536 full; do for K in 10 40; do
  CDIR=/scratch/yorguin/jax_cache/sa_${CH}_$K; rm -rf $CDIR; mkdir -p $CDIR
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CH=$CH K=$K \
  JAX_COMPILATION_CACHE_DIR=$CDIR PYTHONPATH=/scratch/yorguin/amica_main_src:$PYTHONPATH \
  $R/.venv_fir_gpu/bin/python - <<'PY'
import os,time,numpy as np,jax
ch=os.environ["CH"]; K=int(os.environ["K"]); cs=None if ch=="full" else int(ch)
from amica import Amica,AmicaConfig
X=np.random.default_rng(0).standard_normal((64,200000)); X=X-X.mean(1,keepdims=True)
cfg=AmicaConfig(max_iter=K,num_mix_comps=3,lrate=0.1,do_newton=True,do_sphere=False,do_mean=False,chunk_size=cs)
m=Amica(cfg,random_state=0)
t=time.perf_counter(); r=m.fit(X); jax.block_until_ready(getattr(r,"W",getattr(r,"unmixing_",r))); dt=time.perf_counter()-t
open(os.environ["MEAS"],"a").write("amica %s %d %.4f\n"%(ch,K,dt))
PY
done; done

# ---------- scott + pyamica (main, torch cuda) ----------
source $R/.venv_competitors_main/bin/activate
for CH in 1024 4096 16384 65536 full; do for K in 10 40; do
  CH=$CH K=$K python - <<'PY'
import os,time,numpy as np,torch
ch=os.environ["CH"]; K=int(os.environ["K"]); bs=None if ch=="full" else int(ch)
X=np.random.default_rng(0).standard_normal((64,200000)); X=X-X.mean(1,keepdims=True)
from amica import AMICA as S
kw={} if bs is None else {"batch_size":bs}
m=S(n_components=64,n_mixtures=3,device="cuda",n_models=1,mean_center=False,whiten=None,max_iter=K,do_newton=True,newt_start=50,random_state=0,verbose=0,**kw)
t=time.perf_counter(); m.fit(X.T); torch.cuda.synchronize(); dt=time.perf_counter()-t
open(os.environ["MEAS"],"a").write("scott %s %d %.4f\n"%(ch,K,dt))
PY
done; done
for CH in 1024 4096 16384 65536 full; do for K in 10 40; do
  CH=$CH K=$K python - <<'PY'
import os,time,numpy as np,torch,pyamica
ch=os.environ["CH"]; K=int(os.environ["K"]); ct=None if ch=="full" else int(ch)
X=np.random.default_rng(0).standard_normal((64,200000)); X=X-X.mean(1,keepdims=True)
m=pyamica.AMICA(n_components=64,n_models=1,n_mix=3,max_iter=K,lrate=0.1,rho0=1.5,do_sphere=False,do_newton=True,newt_start=50,device="cuda",dtype=torch.float64,fix_init=True,chunk_t=ct)
Xt=torch.from_numpy(X.T).to("cuda")
t=time.perf_counter(); m.fit(Xt); torch.cuda.synchronize(); dt=time.perf_counter()-t
open(os.environ["MEAS"],"a").write("pyamica %s %d %.4f\n"%(ch,K,dt))
PY
done; done
deactivate

# ---------- pAMICA (main, torch cuda, py3.12) ----------
module load python/3.12 >/dev/null 2>&1
source $R/.venv_pamica_main/bin/activate
for CH in 512 4096 16384 65536 full; do for K in 10 40; do
  CH=$CH K=$K python - <<'PY'
import os,time,numpy as np,torch
from pamica import AMICA
ch=os.environ["CH"]; K=int(os.environ["K"]); bs=200000 if ch=="full" else int(ch)
X=np.random.default_rng(0).standard_normal((64,200000)); X=X-X.mean(1,keepdims=True)
m=AMICA(n_models=1,n_mix=3,device="cuda",verbose=False)
t=time.perf_counter(); m.fit(X,max_iter=K,lrate=0.1,do_mean=False,do_sphere=True,do_newton=True,block_size=bs); torch.cuda.synchronize(); dt=time.perf_counter()-t
open(os.environ["MEAS"],"a").write("pamica %s %d %.4f\n"%(ch,K,dt))
PY
done; done
deactivate

# ---------- compute steady ms/it + fixed overhead ----------
module load python/3.11 >/dev/null 2>&1
python3 - <<'PY'
import collections
d=collections.defaultdict(dict)
for ln in open("/scratch/yorguin/steady_meas.txt"):
    p=ln.split();
    if len(p)==4: d[(p[0],p[1])][int(p[2])]=float(p[3])
print("\n=== STEADY-STATE ms/iter (differenced) + fixed overhead ===")
print("  impl      chunk     T10    T40   steady_ms/it   fixed_s")
for (impl,ch),v in sorted(d.items()):
    if 10 in v and 40 in v:
        steady=(v[40]-v[10])/30*1000; fixed=v[10]-10*steady/1000
        print("  %-9s %-7s %6.2f %6.2f   %8.1f      %6.2f"%(impl,ch,v[10],v[40],steady,fixed))
PY
echo STEADY_ALL_DONE
