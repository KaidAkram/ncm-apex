"""
NCM-Apex: Checkpoint & Benchmark Dataset Download CLI
Downloads and verifies official checkpoints from Hugging Face:
- Sudoku (easy, medium, hard)
- TinyStories
- WMT14
"""

import argparse
import os
import sys

from src.utils.checkpoint_loader import (
    download_official_checkpoint,
    download_official_dataset,
    load_sudoku_ncm_model,
)


def main():
    parser = argparse.ArgumentParser(description="Download NCM official checkpoints & datasets")
    parser.add_argument("--tasks", nargs="+", default=["sudoku"], choices=["sudoku", "tinystories", "wmt14", "all"])
    parser.add_argument("--tiers", nargs="+", default=["easy", "medium", "hard"], choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    tasks = ["sudoku", "tinystories", "wmt14"] if "all" in args.tasks else args.tasks

    for task in tasks:
        print(f"\n==========================================")
        print(f"Ingesting official task: {task.upper()}")
        print(f"==========================================")
        if task == "sudoku":
            for tier in args.tiers:
                print(f"\nDownloading Sudoku [{tier}]...")
                ck_files = download_official_checkpoint(task="sudoku", tier=tier)
                print(f"  Tokenizer: {ck_files['tokenizer']}")
                print(f"  Generator: {ck_files['generator']}")
                dt_files = download_official_dataset(task="sudoku", tier=tier)
                print(f"  Puzzles:   {dt_files['puzzles']}")
                print(f"  Solutions: {dt_files['solutions']}")
        else:
            print(f"\nDownloading {task}...")
            ck_files = download_official_checkpoint(task=task)
            print(f"  Tokenizer: {ck_files['tokenizer']}")
            print(f"  Generator: {ck_files['generator']}")
            dt_files = download_official_dataset(task=task)
            print(f"  Dataset:   {dt_files}")

    print("\n[SUCCESS] Checkpoint download and ingestion complete!")


if __name__ == "__main__":
    main()
