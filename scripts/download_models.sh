#!/usr/bin/env bash
# Download the dense Qwen2.5-7B-Instruct model for the v3 pruning sweep.
#
# Usage: bash download_models.sh [repo_id ...]
#   With no args, downloads only the pinned dense Qwen2.5-7B-Instruct model.
#
# Note: LLaMA3.1-8B (paper's flagship) was dropped by request; Qwen2.5 has
# no 8B text size, so Qwen2.5-7B-Instruct is used as the closest stand-in.

set -euo pipefail

VENV=/home/kailais/venvs/hfdl

if [ ! -x "$VENV/bin/hf" ]; then
  uv venv "$VENV"
  uv pip install --python "$VENV/bin/python" huggingface_hub
fi

# Xet high-performance transfer replaces the deprecated HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    Qwen/Qwen2.5-7B-Instruct
  )
fi

for m in "${MODELS[@]}"; do
  printf '[%s] downloading %s\n' "$(date -u +%FT%TZ)" "$m"
  case "$m" in
    Qwen/Qwen2.5-7B-Instruct) revision=a09a35458c702b33eeacc393d103063234e8bc28 ;;
    *) revision=main ;;
  esac
  "$VENV/bin/hf" download "$m" --revision "$revision"
done

printf '[%s] ALL_DOWNLOADS_SUCCEEDED\n' "$(date -u +%FT%TZ)"
