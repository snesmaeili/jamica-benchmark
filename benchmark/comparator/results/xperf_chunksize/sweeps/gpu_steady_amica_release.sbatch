#!/bin/bash
#SBATCH --job-name=amica_steady
#SBATCH --account=def-kjerbi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/yorguin/trillium_out/%x-%j.out
#SBATCH --error=/scratch/yorguin/trillium_out/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 python/3.11 scipy-stack/2026a cuda/12.6 cudnn
[ -n "$CUDA_HOME" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
source /scratch/yorguin/amica-benchmark-repro/.venv_fir_gpu/bin/activate
echo "=== amica RELEASE 92003b4 GPU steady-state: compile split out, +512/1024, cold cache, 30 iters, 64x200k ==="
echo "    (panel fix H2/H5: report first-iter-ms vs steady median ms/iter separately)"
for CS in 512 1024 4096 16384 65536 full; do
  CDIR=/scratch/yorguin/jax_cache/ast_$CS; rm -rf $CDIR; mkdir -p $CDIR   # cold cache per point
  AMICA_CS=$CS JAX_COMPILATION_CACHE_DIR=$CDIR python - <<'PY'
import os, time, numpy as np, jax
cs_env=os.environ["AMICA_CS"]; cs=None if cs_env=="full" else int(cs_env)
from amica import Amica, AmicaConfig
nc,ns,it=64,200000,30
X=np.random.default_rng(0).standard_normal((nc,ns)); X=X-X.mean(1,keepdims=True)
cfg=AmicaConfig(max_iter=it,num_mix_comps=3,lrate=0.1,do_newton=True,do_sphere=False,do_mean=False,chunk_size=cs)
m=Amica(cfg,random_state=0)
t0=time.perf_counter(); r=m.fit(X); jax.block_until_ready(getattr(r,"W",getattr(r,"unmixing_",r)))
total=time.perf_counter()-t0
# per-iteration times if exposed by the fit
itimes=None
for obj in (r,m):
    for attr in ("iteration_times","iter_times","_iteration_times"):
        v=getattr(obj,attr,None)
        if v is not None and len(v)>=it-2: itimes=np.asarray(v,dtype=float); break
    if itimes is not None: break
vram=0.0
try: vram=max(d.memory_stats().get("peak_bytes_in_use",0) for d in jax.devices())/1e9
except Exception: pass
lbl="full" if cs is None else str(cs)
if itimes is not None:
    ms=itimes*1000.0 if itimes.max()<50 else itimes  # normalize s->ms if needed
    first=ms[0]; steady=np.median(ms[5:])   # drop 5 warmup
    print("  chunk=%-7s first_iter=%7.0fms  steady_median=%6.1fms/it  total=%5.1fs  VRAM=%4.2fGB  (n_it=%d)"%(lbl,first,steady,total,vram,len(ms)))
else:
    print("  chunk=%-7s (no iteration_times exposed) total=%5.1fs (%5.0f ms/it avg) VRAM=%4.2fGB"%(lbl,total,total/it*1000,vram))
PY
done
echo AMICA_STEADY_DONE
