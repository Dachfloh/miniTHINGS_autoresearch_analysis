#!/usr/bin/env python3
"""Linear regression: predict an experiment metric from a public benchmark score.

Joins benchmarks/benchmarks.csv (public scores, wide: one row per benchmark,
one column per model) with ../experiment_log.csv (internal metrics, one row
per model/tag) on model name, then fits metric ~ benchmark with ordinary
least squares.

Usage:
    python regress.py --benchmark "Terminal-Bench 2.1" --metric auc
    python regress.py --benchmarks benchmarks_2.csv --benchmark "AutomationBench" --metric mean_acc
    python regress.py --benchmark "DeepSWE v1.1" --metric mean_acc --tag sep4
    python regress.py --benchmark "Terminal-Bench 2.1" --metric std --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_BENCH_CSV = BENCH_DIR / "benchmarks.csv"
LOG_CSV = BENCH_DIR.parent / "experiment_log.csv"

AG_COLS = ["ag1", "ag2", "ag3", "ag4", "ag5"]


def metric_column(df: pd.DataFrame, metric: str):
    """Return the metric values for each (model, tag) row of the log."""
    if metric == "max_acc":
        return df[AG_COLS].max(axis=1)
    if metric in df.columns:
        return df[metric]
    raise SystemExit(
        f"Unknown metric '{metric}'. Choose from: mean_acc, max_acc, std, auc, "
        + ", ".join(AG_COLS)
    )


def load_data(bench_csv: Path, benchmark: str, metric: str,
              tag: str | None) -> pd.DataFrame:
    log = pd.read_csv(LOG_CSV)
    bench = pd.read_csv(bench_csv, index_col=0)

    if benchmark not in bench.index:
        raise SystemExit(
            f"Unknown benchmark '{benchmark}'. Available: " + ", ".join(bench.index)
        )
    if tag:
        log = log[log["tag"] == tag]
        if log.empty:
            raise SystemExit(f"No rows with tag '{tag}' in {LOG_CSV}")

    # One row per model: benchmark score from the wide CSV, metric from the log.
    points = log.assign(metric=metric_column(log, metric)).rename(
        columns={"model": "model", "metric": "metric"}
    )[["model", "tag", "metric"]].copy()
    points["benchmark"] = points["model"].map(bench.loc[benchmark])

    missing = points[points["benchmark"].isna()]
    if not missing.empty:
        print(
            f"Dropping models without a {benchmark} score: "
            + ", ".join(missing["model"]),
            file=sys.stderr,
        )
    unlogged = [m for m in bench.columns if m not in set(log["model"])]
    if unlogged:
        print(
            f"Note: no logged experiments for: " + ", ".join(unlogged),
            file=sys.stderr,
        )
    return points.dropna(subset=["benchmark", "metric"])


def fit(points: pd.DataFrame):
    """OLS fit; returns (slope, intercept, r2, pvalue-or-None)."""
    x = points["benchmark"].to_numpy()
    y = points["metric"].to_numpy()
    n = len(x)
    if n < 2:
        raise SystemExit("Need at least 2 points to fit a line.")

    try:
        from scipy import stats

        res = stats.linregress(x, y)
        return res.slope, res.intercept, res.rvalue**2, res.pvalue
    except ImportError:
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return slope, intercept, r2, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", metavar="CSV", default=str(DEFAULT_BENCH_CSV),
                        help=f"Benchmark scores CSV (default: {DEFAULT_BENCH_CSV.name} "
                             "in this script's directory)")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark row from the benchmarks CSV, e.g. 'Terminal-Bench 2.1'")
    parser.add_argument("--metric", required=True,
                        help="Column of experiment_log.csv: mean_acc, max_acc, std, auc, ag1..ag5")
    parser.add_argument("--tag", default=None,
                        help="Restrict to one run tag (e.g. sep4); default: all rows")
    parser.add_argument("--plot", metavar="OUT",
                        default=None,
                        help="Save a scatter plot with the fitted line to this path")
    args = parser.parse_args()

    bench_csv = Path(args.benchmarks)
    if not bench_csv.is_absolute():
        bench_csv = BENCH_DIR / bench_csv
    if not bench_csv.exists():
        raise SystemExit(f"Benchmarks CSV not found: {bench_csv}")

    points = load_data(bench_csv, args.benchmark, args.metric, args.tag)
    print(f"\n{args.benchmark} -> {args.metric}"
          + (f" (tag {args.tag})" if args.tag else ""))
    print(points.to_string(index=False))

    slope, intercept, r2, p = fit(points)
    print(f"\nn={len(points)}")
    print(f"metric = {slope:.6f} * benchmark + {intercept:.6f}")
    print(f"R^2 = {r2:.4f}")
    if p is not None:
        print(f"p-value = {p:.4f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(points["benchmark"], points["metric"])
        xs = np.linspace(points["benchmark"].min(), points["benchmark"].max(), 10)
        ax.plot(xs, slope * xs + intercept, "--")
        for _, row in points.iterrows():
            ax.annotate(row["model"], (row["benchmark"], row["metric"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(args.benchmark)
        ax.set_ylabel(args.metric)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()