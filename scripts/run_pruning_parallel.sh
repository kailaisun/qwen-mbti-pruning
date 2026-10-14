#!/usr/bin/env bash
# Two independent workers on one sufficiently free H200, or separate GPUs.
# Expected peak allocations: ~24 GiB SparseGPT + ~18 GiB other methods.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUTDIR=${OUTDIR:-outputs/pruning_v3}
LOGDIR=${LOGDIR:-logs/pruning_v3}
GPU_SPARSE=${GPU_SPARSE:-2}
GPU_OTHER=${GPU_OTHER:-2}
mkdir -p "$LOGDIR"
mapfile -t sparse_tags < <("$PY" -B -c "import sys; sys.path.insert(0,'src'); from run_pruning_sweep import experiment_matrix; print('\n'.join(c['tag'] for c in experiment_matrix() if c['method']=='sparsegpt'))")
mapfile -t other_tags < <("$PY" -B -c "import sys; sys.path.insert(0,'src'); from run_pruning_sweep import experiment_matrix; print('\n'.join(c['tag'] for c in experiment_matrix() if c['method']!='sparsegpt'))")
if [ "${#sparse_tags[@]}" -ne 12 ] || [ "${#other_tags[@]}" -ne 52 ]; then
  printf 'Invalid experiment partition.\n' >&2
  exit 1
fi
for worker_log in sparse_worker.log other_worker.log; do
  if [ -f "$LOGDIR/$worker_log" ]; then
    mv "$LOGDIR/$worker_log" "$LOGDIR/${worker_log%.log}.before-retry.$(date -u +%Y%m%dT%H%M%SZ).$$.log"
  fi
done
CUDA_VISIBLE_DEVICES="$GPU_SPARSE" PYTHONUNBUFFERED=1 "$PY" -B src/run_pruning_sweep.py \
  --outdir "$OUTDIR" --tags "${sparse_tags[@]}" > "$LOGDIR/sparse_worker.log" 2>&1 & sparse_pid=$!
CUDA_VISIBLE_DEVICES="$GPU_OTHER" PYTHONUNBUFFERED=1 "$PY" -B src/run_pruning_sweep.py \
  --outdir "$OUTDIR" --tags "${other_tags[@]}" > "$LOGDIR/other_worker.log" 2>&1 & other_pid=$!
failed=0
for worker in "$sparse_pid" "$other_pid"; do
  if ! wait "$worker"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
  printf 'A worker failed; retain results and inspect worker logs.\n' >&2
  exit 1
fi
"$PY" -B src/analyze_pruning_sweep.py --outdir "$OUTDIR"
printf 'ALL_PRUNING_RUNS_VERIFIED_AND_ANALYZED %s\n' "$(date -u +%FT%TZ)"
