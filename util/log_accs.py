#!/usr/bin/env python3
"""Log the final val_acc of 5 result files plus the baseline-subtracted mean-curve
AUC (improvement above each run's baseline config) into a summary CSV.

Usage:
    python log_accs.py <model> <tag>
"""

import argparse
import os
import math
import sys
from pathlib import Path

import pandas as pd

# Reuse the per-run "last keep" series from the multi-run plot script, so the
# AUC here follows the same curve (green mean line), minus the baseline.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_multi_results import last_keep_series  # noqa: E402

OUTPUT_CSV = "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch_analysis/experiment_log.csv"
HEADER = ["model", "tag", "ag1", "ag2", "ag3", "ag4", "ag5", "mean_acc", "std", "auc"]

FILES = [
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent1/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent2/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent3/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent4/results.tsv",
    "/home/staff/g/glados/autoresearch/miniTHINGS_autoresearch-agent5/results.tsv",
]

ROUND_DIGITS = 6


def read_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "val_acc" not in df.columns:
        raise ValueError(f"Column 'val_acc' not found in {path}")
    return df


def area_under_mean_curve(dfs: list[pd.DataFrame]) -> float:
    """Sum over experiments of the mean improvement above each run's baseline.

    Mirrors the green curve of plot_multi_results: runs of different lengths
    are compared only over the range every run covers. The "final" test-set
    evaluation row is not part of the curve. Each run's own baseline (its
    first keep) is subtracted before averaging, so the value measures
    accuracy improvement delivered above the starting config — a run that
    never beats its baseline scores ~0, regardless of how strong that
    baseline was.
    """
    curves = []
    for df in dfs:
        if "status" in df.columns:
            df = df[df["status"] != "final"]
        series = last_keep_series(df)
        curves.append(series - series.iloc[0])
    # concat axis=1: one column per run, rows = experiment indices, so
    # dropna removes the tail past the shortest run (not whole runs).
    means = pd.concat(curves, axis=1).dropna(axis=0, how="any").mean(axis=1)
    return float(means.sum())


def log_max_test_acc(model: str, tag: str, csv_paths: list[str], output_csv: str = OUTPUT_CSV) -> None:
    if len(csv_paths) != 5:
        raise ValueError(f"Expected exactly 5 files, got {len(csv_paths)}")

    dfs = [read_results(path) for path in csv_paths]
    # Last row = the "final" test-set evaluation of the best config.
    max_accs = [round(df["val_acc"].iloc[-1], ROUND_DIGITS) for df in dfs]

    mean_acc = round(sum(max_accs) / len(max_accs), ROUND_DIGITS)
    variance = sum((x - mean_acc) ** 2 for x in max_accs) / len(max_accs)
    std = round(math.sqrt(variance), ROUND_DIGITS)
    auc = round(area_under_mean_curve(dfs), ROUND_DIGITS)

    row = [model, tag] + max_accs + [mean_acc, std, auc]

    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    pd.DataFrame([row], columns=HEADER).to_csv(
        output_csv, mode="a", header=write_header, index=False
    )
    print(f"Logged {model}/{tag}: {[f'{a:.6f}' for a in max_accs]}, auc {auc:.6f} -> {output_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model name")
    parser.add_argument("tag", help="Run tag")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV (default: results.csv)")
    args = parser.parse_args()

    log_max_test_acc(args.model, args.tag, FILES, args.output)


if __name__ == "__main__":
    main()