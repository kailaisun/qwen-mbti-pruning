#!/usr/bin/env bash
# Eight disjoint workers; two on each sufficiently free H200 by default.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUTDIR=${OUTDIR:-outputs/gemma4_12b_v3}
LOGDIR=${LOGDIR:-logs/gemma4_12b_v3}
read -r -a gpus <<< "${GEMMA_GPUS:-0 3 4 7}"
if [ "${#gpus[@]}" -ne 4 ]; then
  printf 'Set GEMMA_GPUS to four GPU indices with sufficient free capacity.\n' >&2
  exit 1
fi
mkdir -p "$LOGDIR"
pids=()
for worker in 0 1 2 3 4 5 6 7; do
  mapfile -t tags < <("$PY" -B -c "import sys;sys.path.insert(0,'src');from run_pruning_sweep import experiment_matrix;w=$worker;configs=[c for c in experiment_matrix() if (c['method']=='sparsegpt')==(w<4)];print('\n'.join(c['tag'] for i,c in enumerate(configs) if i%4==w%4))")
  expected=13
  if [ "$worker" -lt 4 ]; then expected=3; fi
  if [ "${#tags[@]}" -ne "$expected" ]; then
    printf 'Invalid partition for worker %s\n' "$worker" >&2
    exit 1
  fi
  gpu=${gpus[$((worker%4))]}
  worker_log="$LOGDIR/worker_$worker.log"
  if [ -f "$worker_log" ]; then
    mv "$worker_log" "$LOGDIR/worker_$worker.before-retry.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" -B src/run_gemma_sweep.py \
    --outdir "$OUTDIR" --tags "${tags[@]}" > "$worker_log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
  printf 'A Gemma worker failed; retain outputs and inspect logs.\n' >&2
  exit 1
fi
"$PY" -B src/analyze_gemma_sweep.py --outdir "$OUTDIR"
printf 'ALL_GEMMA_RUNS_VERIFIED_AND_ANALYZED %s\n' "$(date -u +%FT%TZ)"
