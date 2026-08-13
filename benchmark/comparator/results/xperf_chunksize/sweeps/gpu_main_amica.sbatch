#!/bin/bash
#SBATCH --job-name=gpu_main_amica
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
export PYTHONPATH=/scratch/yorguin/amica_main_src:$PYTHONPATH
source /scratch/yorguin/amica-benchmark-repro/.venv_fir_gpu/bin/activate
python -c "import amica; print('amica main file:', amica.__file__)"
echo "amica-python MAIN (df18b5e) chunk_size sweep, GPU H100, 20 iters, 64x200k"
for CS in full auto 65536 16384 4096; do
  CDIR=/scratch/yorguin/jax_cache/agm_$CS; rm -rf $CDIR; mkdir -p $CDIR
  AMICA_CS=$CS JAX_COMPILATION_CACHE_DIR=$CDIR python - <<'PY'
import os, time, numpy as np, jax
cs_env=os.environ["AMICA_CS"]; cs=None if cs_env=="full" else ("auto" if cs_env=="auto" else int(cs_env))
from amica import Amica, AmicaConfig
nc,ns=64,200000
X=np.random.default_rng(0).standard_normal((nc,ns)); X=X-X.mean(1,keepdims=True)
cfg=AmicaConfig(max_iter=20,num_mix_comps=3,lrate=0.1,do_newton=True,do_sphere=False,do_mean=False,chunk_size=cs)
m=Amica(cfg,random_state=0)
t0=time.perf_counter(); r=m.fit(X); jax.block_until_ready(getattr(r,"W",getattr(r,"unmixing_",r)))
dt=time.perf_counter()-t0
vram=0.0
try: vram=max(d.memory_stats().get("peak_bytes_in_use",0) for d in jax.devices())/1e9
except Exception: pass
print("  chunk_size=%-8s %6.1fs (%5.0f ms/it) VRAM=%5.2f GB"%(cs_env,dt,dt/20*1000,vram))
PY
done
echo GPU_MAIN_AMICA_DONE
