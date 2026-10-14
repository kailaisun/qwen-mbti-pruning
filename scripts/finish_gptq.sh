#!/usr/bin/env bash
# Retry only missing corrected GPTQ results; never mix legacy protocol outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-7}

PY=/home/kailais/venvs/mbti/bin/python
OUTDIR=${OUTDIR:-outputs/corrected_v2}
mkdir -p logs/corrected_v2
if [ ! -f "$OUTDIR/qwen25-7b-gptq4/summary.json" ]; then
  "$PY" -B src/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --revision e9c932ac1893a49ae0fc497ad6e1e86e2e39af20 --local-files-only \
    --outdir "$OUTDIR" --tag qwen25-7b-gptq4 > logs/corrected_v2/gptq_retry.log 2>&1
fi
"$PY" -B src/analyze_results.py --outdir "$OUTDIR" --legacy-outdir outputs
