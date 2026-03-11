#!/bin/bash
# Download Nemotron pretraining datasets in parallel (one process per subset).
# Each process gets a unique tqdm position so progress bars don't overlap.
# Usage: bash tools/download_all_nemotron.sh

set -e

source .venv/bin/activate
export HF_HOME=/mnt/hf_cache

mkdir -p /mnt/pretrain_data/nemotron_code_v2/{train,val}
mkdir -p /mnt/pretrain_data/nemotron_specialized_v1/{train,val}

PIDS=()
POS=0

# --- Nemotron-Pretraining-Code-v2 (skip Synthetic-Question-Answering: 390M rows) ---
for subset in "Synthetic-Student-Teacher" "Synthetic-Code-Review" "Synthetic-Rewriting" "Synthetic-Transpilation"; do
    python tools/download_nemotron_data.py \
        --dataset nvidia/Nemotron-Pretraining-Code-v2 \
        --output-dir /mnt/pretrain_data/nemotron_code_v2 \
        --subsets "$subset" \
        --tqdm-position $POS &
    PIDS+=($!)
    POS=$((POS + 1))
done

# --- Nemotron-Pretraining-Specialized-v1 ---
for subset in "Nemotron-Pretraining-Wiki-Rewrite" "Nemotron-Pretraining-Math-Textbooks" "Nemotron-Pretraining-STEM-SFT" "Nemotron-Pretraining-Scientific-Coding" "Nemotron-Pretraining-RQA" "Nemotron-Pretraining-InfiniByte-Reasoning"; do
    python tools/download_nemotron_data.py \
        --dataset nvidia/Nemotron-Pretraining-Specialized-v1 \
        --output-dir /mnt/pretrain_data/nemotron_specialized_v1 \
        --subsets "$subset" \
        --tqdm-position $POS &
    PIDS+=($!)
    POS=$((POS + 1))
done

echo "Launched $POS download processes. PIDs: ${PIDS[*]}"
echo "Ctrl+C to kill all."

trap 'echo "Killing all downloads..."; kill "${PIDS[@]}" 2>/dev/null; exit 1' INT

wait
echo "All downloads complete."
