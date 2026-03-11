"""Download Nemotron pretraining datasets and save as train/val parquet splits.

Uses streaming to avoid downloading the full dataset to cache. Takes the first
--fraction of each subset, then stops. Subsets are downloaded in parallel using
threads (I/O bound).

Supported datasets:
  - nvidia/Nemotron-CC-v2.1 (text column: "text")
  - nvidia/Nemotron-Pretraining-Code-v2 (text column: "content")
  - nvidia/Nemotron-Pretraining-Specialized-v1 (text column: "text")
  - nvidia/Nemotron-Pretraining-Dataset-sample (text column: "text")
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import dataset_info
from tqdm import tqdm


# Dataset configs: {dataset_name: (subsets, text_column)}
DATASET_CONFIGS = {
    "nvidia/Nemotron-CC-v2.1": {
        "text_column": "text",
        "subsets": [
            "High-Quality",
            "High-Quality-DQA",
            "High-Quality-Synthetic",
            "High-Quality-Translated-To-English",
            "High-Quality-Translated-To-English-Synthetic",
            "Medium-High-Quality",
            "Medium-High-Quality-Synthetic",
            "Medium-High-Quality-Translated-To-English",
            "Medium-Quality",
        ],
    },
    "nvidia/Nemotron-Pretraining-Code-v2": {
        "text_column": "content",
        "subsets": [
            # Skip Nemotron-Code-Metadata (no text, only repo metadata)
            "Synthetic-Question-Answering",
            "Synthetic-Student-Teacher",
            "Synthetic-Code-Review",
            "Synthetic-Rewriting",
            "Synthetic-Transpilation",
        ],
    },
    "nvidia/Nemotron-Pretraining-Specialized-v1": {
        "text_column": "text",
        "subsets": [
            "Nemotron-Pretraining-Wiki-Rewrite",
            "Nemotron-Pretraining-Math-Textbooks",
            "Nemotron-Pretraining-STEM-SFT",
            "Nemotron-Pretraining-Scientific-Coding",
            "Nemotron-Pretraining-RQA",
            "Nemotron-Pretraining-InfiniByte-Reasoning",
        ],
    },
    "nvidia/Nemotron-Pretraining-Dataset-sample": {
        "text_column": "text",
        "subsets": [
            "Nemotron-CC-High-Quality",
            "Nemotron-CC-High-Quality-Synthetic",
            "Nemotron-CC-Diverse-QA",
            "Nemotron-CC-MATH",
            "Nemotron-CC-Translated-Diverse-QA",
            "Nemotron-Synthetic-Code",
        ],
    },
}


def get_subset_row_count(dataset_name, subset):
    """Fetch the total row count for a subset via the HF Hub API (metadata only)."""
    try:
        info = dataset_info(dataset_name, config_name=subset)
        for split_info in info.splits.values():
            if split_info.name == "train":
                return split_info.num_examples
    except Exception:
        pass
    return None


def stream_subset(dataset_name, subset, text_column, fraction, position=0):
    """Stream a subset and return the first fraction of text samples as a list."""
    ds_iter = load_dataset(dataset_name, subset, split="train", streaming=True)

    n_total = get_subset_row_count(dataset_name, subset)
    n_keep = max(1, int(n_total * fraction)) if n_total is not None else None

    texts = []
    iterator = tqdm(ds_iter, total=n_keep, desc=f"  {subset}", position=position,
                    leave=True, dynamic_ncols=True, unit=" rows")
    for row in iterator:
        texts.append(row[text_column])
        if n_keep is not None and len(texts) >= n_keep:
            break
    return texts


def process_subset(dataset_name, subset, text_column, fraction, val_fraction,
                   seed, train_dir, val_dir, position=0):
    """Download, shuffle, split, and write one subset to parquet."""
    texts = stream_subset(dataset_name, subset, text_column, fraction, position=position)

    if len(texts) == 0:
        tqdm.write(f"  WARNING: No rows kept for {subset}, skipping.")
        return subset, 0, 0

    # Shuffle and split into train/val with per-subset deterministic RNG
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(texts))
    n_val = max(1, int(len(texts) * val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]

    # Write parquet files
    train_path = os.path.join(train_dir, f"{subset}.parquet")
    val_path = os.path.join(val_dir, f"{subset}.parquet")

    pq.write_table(pa.table({"text": train_texts}), train_path)
    pq.write_table(pa.table({"text": val_texts}), val_path)

    tqdm.write(f"  {subset}: {len(train_texts)} train, {len(val_texts)} val -> {train_path}")
    return subset, len(train_texts), len(val_texts)


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron pretraining dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="HuggingFace dataset name")
    parser.add_argument("--subsets", type=str, nargs="*", default=None,
                        help="Subsets to download. If not specified, downloads all subsets.")
    parser.add_argument("--skip-subsets", type=str, nargs="*", default=None,
                        help="Subsets to skip (e.g. --skip-subsets Synthetic-Question-Answering)")
    parser.add_argument("--text-column", type=str, default=None,
                        help="Override text column name (auto-detected from dataset config)")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Fraction of each subset to keep (e.g. 0.1 for 10%%)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of kept data to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel download threads")
    parser.add_argument("--tqdm-position", type=int, default=0,
                        help="tqdm bar position (for parallel subprocesses)")
    args = parser.parse_args()

    # Look up dataset config
    ds_config = DATASET_CONFIGS[args.dataset]
    text_column = args.text_column or ds_config["text_column"]
    subsets = args.subsets or ds_config["subsets"]
    if args.skip_subsets:
        subsets = [s for s in subsets if s not in args.skip_subsets]

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"Downloading {len(subsets)} subsets with {args.workers} workers, "
          f"keeping first {args.fraction*100:.0f}%...")

    total_train = 0
    total_val = 0

    if args.workers <= 1:
        for i, subset in enumerate(subsets):
            subset_seed = args.seed + i
            try:
                _, n_train, n_val = process_subset(
                    args.dataset, subset, text_column, args.fraction,
                    args.val_fraction, subset_seed, train_dir, val_dir,
                    position=args.tqdm_position,
                )
                total_train += n_train
                total_val += n_val
            except Exception as e:
                print(f"\n  ERROR on {subset}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, subset in enumerate(subsets):
                subset_seed = args.seed + i
                future = executor.submit(
                    process_subset,
                    args.dataset, subset, text_column, args.fraction,
                    args.val_fraction, subset_seed, train_dir, val_dir,
                    position=i,
                )
                futures[future] = subset

            for future in as_completed(futures):
                subset_name = futures[future]
                try:
                    _, n_train, n_val = future.result()
                    total_train += n_train
                    total_val += n_val
                except Exception as e:
                    print(f"\n  ERROR on {subset_name}: {e}")

    print(f"\nDone. {total_train} train, {total_val} val total.")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")


if __name__ == "__main__":
    main()
