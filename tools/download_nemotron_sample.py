"""Download nvidia/Nemotron-Pretraining-Dataset-sample and save as train/val parquet splits."""

import argparse
import os

from datasets import load_dataset


SUBSETS = [
    "Nemotron-CC-High-Quality",
    "Nemotron-CC-High-Quality-Synthetic",
    "Nemotron-CC-Diverse-QA",
    "Nemotron-CC-MATH",
    "Nemotron-CC-Translated-Diverse-QA",
    "Nemotron-Synthetic-Code",
]


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron pretraining sample dataset")
    parser.add_argument("--output-dir", type=str, default="/mnt/pretrain_data/nemotron_sample")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for subset in SUBSETS:
        print(f"Downloading {subset}...")
        ds = load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", subset, split="train")

        # Keep only the text column
        ds = ds.select_columns(["text"])

        # Split into train/val
        split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)

        train_path = os.path.join(train_dir, f"{subset}.parquet")
        val_path = os.path.join(val_dir, f"{subset}.parquet")

        split["train"].to_parquet(train_path)
        split["test"].to_parquet(val_path)

        print(f"  {subset}: {len(split['train'])} train, {len(split['test'])} val rows -> {train_path}")

    print(f"\nDone. Train: {train_dir}, Val: {val_dir}")


if __name__ == "__main__":
    main()
