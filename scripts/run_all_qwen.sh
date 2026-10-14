#!/usr/bin/env bash
# Corrected protocol; fail on any failed run, retain one log per variant.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/kailais/venvs/mbti/bin/python}
GPU=${GPU:-7}
OUTDIR=${OUTDIR:-outputs/corrected_v2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/kailais/mbti_checkpoints/corrected_v2}
LOGDIR=${LOGDIR:-logs/corrected_v2}
mkdir -p "$LOGDIR"
FP_REV=a09a35458c702b33eeacc393d103063234e8bc28
AWQ_REV=b25037543e9394b818fdfca67ab2a00ecc7dd641
GPTQ_REV=e9c932ac1893a49ae0fc497ad6e1e86e2e39af20

run_variant() {
  local gpu=$1 tag=$2 model=$3 revision=$4
  shift 4
  if [ -f "$OUTDIR/$tag/summary.json" ]; then
    "$PY" -B - "$OUTDIR/$tag/summary.json" "$model" "$revision" <<'PY'
import hashlib, json, sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())
assert s.get('protocol_version')=='2.0-native-final-logits' and s['n_items']==60
assert s['model']==sys.argv[2] and s['provenance']['requested_revision']==sys.argv[3]
for name in ['src/mbti_common.py','src/run_experiment.py','data/mbti_items.json']:
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==s['provenance']['source_sha256'][name], name
if s.get('prune'):
    assert s['checkpoint']['reload_verified_items']==60
    assert (Path(s['checkpoint']['path'])/'config.json').is_file()
PY
    printf 'Reusing verified completed result: %s\n' "$tag"
    return
  fi
  if [ -f "$LOGDIR/$tag.log" ]; then
    mv "$LOGDIR/$tag.log" "$LOGDIR/$tag.before-retry.$(date +%s).log"
  fi
  printf '[%s] start %s on GPU %s\n' "$(date -u +%FT%TZ)" "$tag" "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" -B src/run_experiment.py \
    --model "$model" --revision "$revision" --local-files-only \
    --tag "$tag" --outdir "$OUTDIR" "$@" > "$LOGDIR/$tag.log" 2>&1
  printf '[%s] completed %s\n' "$(date -u +%FT%TZ)" "$tag"
}

(
  run_variant "${GPU_DENSE:-$GPU}" qwen25-7b-fp16 Qwen/Qwen2.5-7B-Instruct "$FP_REV"
  run_variant "${GPU_DENSE:-$GPU}" qwen25-7b-prune-u50 Qwen/Qwen2.5-7B-Instruct "$FP_REV" \
    --prune-pattern unstructured --prune-ratio 0.5 --save-pruned-dir "$CHECKPOINT_DIR/qwen25-7b-prune-u50"
  run_variant "${GPU_DENSE:-$GPU}" qwen25-7b-prune-2to4 Qwen/Qwen2.5-7B-Instruct "$FP_REV" \
    --prune-pattern 2to4 --save-pruned-dir "$CHECKPOINT_DIR/qwen25-7b-prune-2to4"
) & dense_pid=$!
run_variant "${GPU_AWQ:-$GPU}" qwen25-7b-awq4 Qwen/Qwen2.5-7B-Instruct-AWQ "$AWQ_REV" & awq_pid=$!
run_variant "${GPU_GPTQ:-$GPU}" qwen25-7b-gptq4 Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 "$GPTQ_REV" & gptq_pid=$!

failed=0
for run_pid in "$dense_pid" "$awq_pid" "$gptq_pid"; do
  if ! wait "$run_pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then
  printf 'At least one run failed; inspect %s.\n' "$LOGDIR" >&2
  exit 1
fi
"$PY" -B src/analyze_results.py --outdir "$OUTDIR" --legacy-outdir outputs
printf 'ALL_RUNS_VERIFIED_AND_ANALYZED %s\n' "$(date -u +%FT%TZ)"
