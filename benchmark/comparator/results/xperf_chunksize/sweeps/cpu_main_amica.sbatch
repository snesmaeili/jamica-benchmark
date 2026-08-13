#!/bin/bash
#SBATCH --job-name=amica_main_cpu
#SBATCH --account=rrg-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/yorguin/%x-%j.out
#SBATCH --error=/scratch/yorguin/%x-%j.err
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
R=/scratch/yorguin/amica-benchmark-repro
export NC=64 NS=100000 IT=10
module load StdEnv/2023 python/3.11 scipy-stack/2026a >/dev/null 2>&1
echo "=== amica MAIN (df18b5e, incl 2cd81e4 CPU rework) chunk_size sweep, CPU, 64x100000, 10 iters, 8 cores ==="
for CS in full auto 65536 16384 4096; do
  CDIR=/scratch/yorguin/jax_cache/amc2_$CS; rm -rf $CDIR; mkdir -p $CDIR
  JAX_PLATFORMS=cpu CS=$CS JAX_COMPILATION_CACHE_DIR=$CDIR PYTHONPATH=/scratch/yorguin/amica_main_src:$PYTHONPATH $R/.venv_fir_gpu/bin/python - <<'PY'
import os,time,resource,numpy as np
cs_env=os.environ["CS"]; cs=None if cs_env=="full" else ("auto" if cs_env=="auto" else int(cs_env))
import amica as _a; from amica import Amica,AmicaConfig
nc,ns,it=int(os.environ["NC"]),int(os.environ["NS"]),int(os.environ["IT"])
X=np.random.default_rng(0).standard_normal((nc,ns));X=X-X.mean(1,keepdims=True)
cfg=AmicaConfig(max_iter=it,num_mix_comps=3,lrate=0.1,do_newton=True,do_sphere=False,do_mean=False,chunk_size=cs)
m=Amica(cfg,random_state=0)
t=time.perf_counter();m.fit(X);d=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
print("   chunk_size=%-8s %6.1fs (%5.0f ms/it) RSS=%5.2f GB   [%s]"%(cs_env,d,d/it*1000,rss,_a.__file__.split("site-packages/")[-1] if "site-packages" in _a.__file__ else "main_src"))
PY
done
echo AMICA_MAIN_CPU_DONE
