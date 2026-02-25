#!/bin/bash
set -e -m

# Preprocess all downloaded Nemotron datasets into Megatron format.
# Runs each subset as a separate background process for maximum parallelism.

source .venv/bin/activate

HF_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TOKENIZER_DIR="/tmp/nemotron_tokenizer"
TOTAL_CPUS=$(nproc)

# Download tokenizer once to avoid 429 rate limits
if [ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]; then
    echo "Downloading tokenizer to $TOKENIZER_DIR..."
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$HF_MODEL').save_pretrained('$TOKENIZER_DIR')"
fi
MODEL="$TOKENIZER_DIR"

# Collect all (input_file, output_dir, output_prefix, text_column) tuples
declare -a JOBS=()

add_jobs() {
    local raw_dir="$1"
    local out_dir="$2"
    local text_col="$3"

    for split in train val; do
        local split_raw="$raw_dir/$split"
        local split_out="$out_dir/$split"
        if [ ! -d "$split_raw" ]; then
            echo "Skipping $split_raw (not found)"
            continue
        fi
        mkdir -p "$split_out"
        for parquet in "$split_raw"/*.parquet; do
            if [ ! -f "$parquet" ]; then
                continue
            fi
            local base=$(basename "$parquet" .parquet)
            JOBS+=("$parquet|$split_out|${base}|$text_col")
        done
    done
}

# CC-v2.1: text column = "text"
add_jobs "/mnt/pretrain_data/nemotron_cc_v2.1" \
         "/mnt/pretrain_data/megatron/nemotron_cc_v2.1" \
         "text"

# Code-v2: download script saves as "text"
add_jobs "/mnt/pretrain_data/nemotron_code_v2" \
         "/mnt/pretrain_data/megatron/nemotron_code_v2" \
         "text"

# Specialized-v1: text column = "text"
add_jobs "/mnt/pretrain_data/nemotron_specialized_v1" \
         "/mnt/pretrain_data/megatron/nemotron_specialized_v1" \
         "text"

NUM_JOBS=${#JOBS[@]}
if [ "$NUM_JOBS" -eq 0 ]; then
    echo "No parquet files found to process."
    exit 0
fi

WORKERS_PER_JOB=$(( TOTAL_CPUS / NUM_JOBS ))
if [ "$WORKERS_PER_JOB" -lt 1 ]; then
    WORKERS_PER_JOB=1
fi

echo "Found $NUM_JOBS files to process with $TOTAL_CPUS CPUs ($WORKERS_PER_JOB workers per job)"
echo ""

PIDS=()
POS=0

for job in "${JOBS[@]}"; do
    IFS='|' read -r input_file output_dir output_prefix text_col <<< "$job"

    echo "[START] $(basename "$input_file") -> ${output_dir}/${output_prefix} (workers=$WORKERS_PER_JOB)"

    python tools/preprocess_megatron_dataset.py \
        --input "$input_file" \
        --output-prefix "$output_prefix" \
        --output-path "$output_dir" \
        --pretrained-model-name-or-path "$MODEL" \
        --text-column "$text_col" \
        --workers "$WORKERS_PER_JOB" \
        --append-eod \
        --tqdm-position "$POS" \
        &

    PIDS+=($!)
    POS=$((POS + 1))
done

echo ""
echo "Launched ${#PIDS[@]} preprocessing jobs. Waiting for completion..."

cleanup() {
    echo "Killing all jobs..."
    for pid in "${PIDS[@]}"; do
        pkill -TERM -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    exit 1
}
trap cleanup INT TERM

# Poll all PIDs; if any exits with error, kill the rest
while true; do
    ALL_DONE=true
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        if [ -z "$pid" ]; then
            continue  # already handled
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished, check exit code
            wait "$pid" 2>/dev/null
            code=$?
            if [ "$code" -ne 0 ]; then
                echo "[FAIL] PID $pid exited with code $code"
                PIDS[$i]=""  # mark handled
                cleanup
            else
                echo "[DONE] PID $pid finished successfully"
                PIDS[$i]=""  # mark handled
            fi
        else
            ALL_DONE=false
        fi
    done
    if $ALL_DONE; then
        break
    fi
    sleep 2
done

echo ""
echo "All jobs completed successfully!"
