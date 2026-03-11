#!/usr/bin/env bash
# LR sweep: 500 steps each, LRs from 5e-5 to 2e-3
# Updates optimizer.lr, lr_scheduler.{max_lr,min_lr,warmup}, step_scheduler.max_steps
#
# Usage:
#   ./tools/lr_sweep_500steps.sh
#   NPROC_PER_NODE=8 ./tools/lr_sweep_500steps.sh
#
# Run from repo root, or set AUTOMODEL_ROOT to the repo path.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AUTOMODEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

CONFIG="examples/llm_pretrain/nemotron_nano_v3_pretrain.yaml"
NPROC="${NPROC_PER_NODE:-8}"
MAX_STEPS=500
WARMUP_STEPS=50  # 10% of 500

LRS=(5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)

for lr in "${LRS[@]}"; do
  # min_lr = max_lr/100 (same ratio as base config)
  min_lr=$(python3 -c "print(${lr}/100)")
  # wandb run name: sanitize lr (e.g. 5e-5 -> 5e-5, 1e-4 -> 1e-4)
  lr_slug=$(echo "$lr" | sed 's/\./_/g' | sed 's/-/_/g')
  echo "=========================================="
  echo "Running LR=${lr} for ${MAX_STEPS} steps (min_lr=${min_lr})"
  echo "=========================================="
  torchrun --nproc-per-node="$NPROC" examples/llm_pretrain/pretrain.py \
    --config "$CONFIG" \
    --step_scheduler.max_steps="$MAX_STEPS" \
    --optimizer.lr="$lr" \
    --lr_scheduler.max_lr="$lr" \
    --lr_scheduler.min_lr="$min_lr" \
    --lr_scheduler.lr_warmup_steps="$WARMUP_STEPS" \
    --lr_scheduler.lr_decay_steps="$MAX_STEPS" \
    --lr_scheduler.wsd_decay_steps=0 \
    --wandb.name="lrsweep_lr${lr_slug}_500steps"
done

echo "LR sweep complete."
