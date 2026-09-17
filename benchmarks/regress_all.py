#!/usr/bin/env python3
"""Run every metric ~ benchmark regression for one benchmarks CSV, rank by R^2,
and plot the best N as side-by-side subplots.

Iterates over all benchmark rows of the CSV and all requested metrics, joins
each with experiment_log.csv on model name (same join as regress.py), fits OLS,
then ranks the valid fits by R^2 and plots the top N.

Usage:
    python regress_all.py
    python regress_all.py --benchmarks benchmarks_2.csv --metrics mean_acc,auc
    python regress_all.py --tag sep4 --top 5 --out best_fits.png
    python regress_all.py --positive --top 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regress import BENCH_DIR, DEFAULT_BENCH_CSV, LOG_CSV, metric_column, fit

ACCENT = "#3B6EF0"  # scatter marks (single series per panel)
FIT = "#333333"  # fitted line
INK_MUTED = "#666666"


def all_fits(bench: pd.DataFrame, log: pd.DataFrame, metrics: list[str],
              tag: str | None, min_points: int) -> pd.DataFrame:
    """Fit metric ~ benchmark for every (benchmark, metric) pair; rank by R^2."""
    if tag:
        log = log[log["tag"] == tag]
        if log.empty:
            raise SystemExit(f"No rows with tag '{tag}' in {LOG_CSV}")
    bench_scores = {b: bench.loc[b].dropna() for b in bench.index}

    rows = []
    for benchmark, scores in bench_scores.items():
        for metric in metrics:
            log_m = log.assign(metric=metric_column(log, metric))
            points = log_m[log_m["model"].isin(scores.index)][
                ["model", "metric"]].copy()
            points["benchmark"] = points["model"].map(scores).astype(float)
            points = points.dropna(subset=["benchmark", "metric"])
            if len(points) < min_points or points["benchmark"].nunique() < 2:
                continue
            try:
                slope, intercept, r2, p = fit(points)
            except SystemExit:
                continue
            rows.append({
                "benchmark": benchmark, "metric": metric, "n": len(points),
                "R2": r2, "p": p, "slope": slope, "intercept": intercept,
                "points": points,
            })
    return pd.DataFrame(
        rows,
        columns=["benchmark", "metric", "n", "R2", "p", "slope",
                 "intercept", "points"],
    ).sort_values("R2", ascending=False, na_position="last")


def plot_top(results: pd.DataFrame, top: int, out: Path, tag: str | None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = results.head(top)
    n = len(top)
    if n == 0:
        raise SystemExit("No valid regressions to plot.")
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (_, r) in zip(axes[0], top.iterrows()):
        points = r["points"]
        ax.scatter(points["benchmark"], points["metric"], s=40, color=ACCENT,
                   zorder=3)
        xs = np.linspace(points["benchmark"].min(), points["benchmark"].max(), 10)
        ax.plot(xs, r["slope"] * xs + r["intercept"], "--", color=FIT,
                linewidth=1.5, zorder=2)
        for _, row in points.iterrows():
            ax.annotate(row["model"], (row["benchmark"], row["metric"]),
                        fontsize=7, color=INK_MUTED,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{r['benchmark']}\n$R^2$={r['R2']:.3f}, n={r['n']}, "
                     + (f"p={r['p']:.3g}" if r["p"] is not None else "p n/a"),
                     fontsize=10)
        ax.set_xlabel(r["benchmark"])
        ax.set_ylabel(r["metric"])
        ax.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    title = "Best regressions: metric ~ benchmark"
    if tag:
        title += f" (tag {tag})"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    print(f"\nPlot saved to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", metavar="CSV",
                        default=str(DEFAULT_BENCH_CSV),
                        help=f"Benchmark scores CSV (default: {DEFAULT_BENCH_CSV.name})")
    parser.add_argument("--metrics",
                        default="mean_acc,max_acc,std,auc",
                        help="Comma-separated metric columns of experiment_log.csv "
                             "(default: mean_acc,max_acc,std,auc; ag1..ag5 also allowed)")
    parser.add_argument("--tag", default=None,
                        help="Restrict to one run tag (e.g. sep4); default: all rows")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of best fits to plot (default: 3)")
    parser.add_argument("--positive", action="store_true",
                        help="Only consider fits with a positive slope "
                             "(metric rises with benchmark score)")
    parser.add_argument("--min-points", type=int, default=3,
                        help="Minimum number of points per fit (default: 3; "
                             "R^2 is degenerate at n=2)")
    parser.add_argument("--out", metavar="PNG", default=None,
                        help="Output plot path (default: regress_all_<stem>.png "
                             "next to the script)")
    args = parser.parse_args()

    bench_csv = Path(args.benchmarks)
    if not bench_csv.is_absolute():
        bench_csv = BENCH_DIR / bench_csv
    if not bench_csv.exists():
        raise SystemExit(f"Benchmarks CSV not found: {bench_csv}")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    bench = pd.read_csv(bench_csv, index_col=0)
    log = pd.read_csv(LOG_CSV)
    results = all_fits(bench, log, metrics, args.tag, args.min_points)
    if args.positive:
        results = results[results["slope"] > 0].reset_index(drop=True)
        if results.empty:
            raise SystemExit("No valid regression had a positive slope.")

    if results.empty:
        raise SystemExit(
            "No regression had at least "
            f"{args.min_points} models with both a benchmark score and a metric."
        )
    show = results.copy()
    show["p"] = show["p"].map(lambda p: f"{p:.4f}" if p is not None else "-")
    show["R2"] = show["R2"].map("{:.4f}".format)
    print(show[["benchmark", "metric", "n", "R2", "p"]].to_string(index=False))

    out = (Path(args.out) if args.out
           else BENCH_DIR / f"regress_all_{bench_csv.stem}.png")
    plot_top(results, args.top, out, args.tag)


if __name__ == "__main__":
    main()