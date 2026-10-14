#!/usr/bin/env bash
# Start in tmux if desired; completed compatible results are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
GPU=${GPU:-2}
OUTDIR=${OUTDIR:-outputs/pruning_v3}
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" -B src/run_pruning_sweep.py --outdir "$OUTDIR" "$@"
"$PY" -B src/analyze_pruning_sweep.py --outdir "$OUTDIR"
printf 'ALL_PRUNING_RUNS_VERIFIED_AND_ANALYZED %s\n' "$(date -u +%FT%TZ)"
