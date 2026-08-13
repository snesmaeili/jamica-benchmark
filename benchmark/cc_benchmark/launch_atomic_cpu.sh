#!/bin/bash
# Fan out the atomic CPU chunk-size campaign (bleeding-edge / main builds): per subject,
# preprocess ONCE then submit isolated (impl, chunk) cells that depend on it (afterok).
# Each cell fits one implementation at one chunk size, loading the cached projected input.
# Fortran is a separate reference cell (its own binary; no release/main axis).
# Run from benchmark/cc_benchmark/ on fir.  Override SUBJECTS / CHUNKS via env.
set -euo pipefail
cd "$(dirname "$0")"
R=/scratch/yorguin/amica-benchmark-repro
FGPU=$R/.venv_fir_gpu/bin/python; ASRC=/scratch/yorguin/amica_main_src
FBIN=/project/rrg-kjerbi/yorguin/amica_fortran_reference/amica17
CACHE_DIR=/scratch/yorguin/realchunk_cache
OUT=/scratch/yorguin/xperf_realchunk_main/cpu_atomic
SUBJECTS="${SUBJECTS:-1 2 3 4 5}"
CHUNKS="${CHUNKS:-1024 4096 16384 65536 100000000}"
VENVS="AMICA_PYTHON_VENV=$FGPU,COMPETITORS_VENV=$R/.venv_competitors_main/bin/python,PAMICA_VENV=$R/.venv_pamica_main/bin/python,AMICA_SRC=$ASRC"
ALL="jamica jamica_chunked jamica_numpy neuromechanist_numpy pyamica_torch scott_huberty_torch pamica_torch fortran_amica17"
skipcomp(){ local keep=$1 s=""; for k in $ALL; do [ "$k" = "$keep" ] || s="$s $k"; done; echo "${s# }"; }
mkdir -p "$CACHE_DIR" /scratch/yorguin/realchunk_cells
np=0; nc=0
for s in $SUBJECTS; do
  st=$(printf "sub-%02d" "$s"); CACHE=$CACHE_DIR/ds004505_${st}_nc64.npz
  PJ=$(sbatch --job-name=prep_${st} --parsable \
       --export="ALL,AMICA_MEM_SUBJECT=$s,AMICA_MEM_NCOMP=64,AMICA_CACHED_INPUT=$CACHE,$VENVS" \
       submit_preprocess_cpu.sh)
  np=$((np+1))
  for c in $CHUNKS; do
    for keep in jamica_chunked pyamica_torch pamica_torch scott_huberty_torch; do
      tl=03:00:00; [ "$keep" = scott_huberty_torch ] && tl=05:00:00
      E="ALL,AMICA_MEM_SUBJECT=$s,AMICA_MEM_ITER=100,AMICA_MEM_CHUNK=$c,AMICA_SCOTT_BATCH=$c,AMICA_PYAMICA_CHUNK=$c,AMICA_PAMICA_BLOCK_SIZE=$c,AMICA_MEM_TAG=c$c,CELL_SKIP=$(skipcomp "$keep"),AMICA_CACHED_INPUT=$CACHE,AMICA_COMPARATOR_RESULTS=$OUT/c$c,$VENVS"
      sbatch --dependency=afterok:$PJ --job-name=cell_${keep%%_*}_${st}_c$c --time=$tl --parsable --export="$E" submit_cell_cpu.sh >/dev/null
      nc=$((nc+1))
    done
    E="ALL,AMICA_MEM_SUBJECT=$s,AMICA_MEM_ITER=100,AMICA_MEM_CHUNK=$c,AMICA_FORTRAN_BLOCK=$c,AMICA_MEM_TAG=c$c,CELL_SKIP=$(skipcomp fortran_amica17),CELL_INCLUDE_FORTRAN=1,AMICA17_BIN=$FBIN,AMICA_CACHED_INPUT=$CACHE,AMICA_COMPARATOR_RESULTS=$OUT/c$c,$VENVS"
    sbatch --dependency=afterok:$PJ --job-name=cell_fortran_${st}_c$c --time=05:00:00 --parsable --export="$E" submit_cell_cpu.sh >/dev/null
    nc=$((nc+1))
  done
done
echo "launched: $np preprocess + $nc cells (subjects: $SUBJECTS)"
