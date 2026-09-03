#!/usr/bin/env python3
"""Log the max test_acc of 7 result files into a summary CSV.

Usage:
    python log_accs.py <model> <tag>
"""

import argparse
import os

import pandas as pd

OUTPUT_CSV = "experiment_log.csv"
HEADER = ["model", "tag", "ag1", "ag2", "ag3", "ag4", "ag5", "ag6", "ag7"]

FILES = [
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent1/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent2/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent3/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent4/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent5/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent6/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent7/results.tsv",
]


def log_max_test_acc(model: str, tag: str, csv_paths: list[str], output_csv: str = OUTPUT_CSV) -> None:
    if len(csv_paths) != 7:
        raise ValueError(f"Expected exactly 7 files, got {len(csv_paths)}")

    max_accs = []
    for path in csv_paths:
        df = pd.read_csv(path, sep="\t")
        if "test_acc" not in df.columns:
            raise ValueError(f"Column 'test_acc' not found in {path}")
        max_accs.append(df["test_acc"].max())

    row = [model, tag] + max_accs

    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    pd.DataFrame([row], columns=HEADER).to_csv(
        output_csv, mode="a", header=write_header, index=False
    )
    print(f"Logged {model}/{tag}: {[f'{a:.4f}' for a in max_accs]} -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name")
    parser.add_argument("tag", help="Run tag")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV (default: results.csv)")
    args = parser.parse_args()

    log_max_test_acc(args.model, args.tag, FILES, args.output)


if __name__ == "__main__":
    main()