#!/usr/bin/env python3
"""Plot mean best-keep val_acc progression across several results.tsv runs.

Each run's value at an experiment index is the val_acc of its most recent
keep (carried forward through discards and crashes). The plot shows the mean
across runs, a shaded band between the best and worst run, and a faint
baseline reference line.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_results import (
    COLOR_BASELINE,
    COLOR_GRID,
    COLOR_KEEP,
    INK_PRIMARY,
    INK_SECONDARY,
    PLOTS_DIR,
    _style_axes,
    parse_tsv,
)

PAST_RUNS_DIR = Path(__file__).parent / "past_runs"
EXPERIMENT_LOG = Path(__file__).parent / "experiment_log.csv"


def last_keep_series(df: pd.DataFrame) -> pd.Series:
    """val_acc of the most recent keep at each experiment index (carried forward)."""
    keep_vals = df["val_acc"].where(df["status"] == "keep")
    return keep_vals.ffill()


def _run_labels(tsv_paths: list[str | Path]) -> list[str]:
    """Unique label per run: the last two path parts (e.g. agent1/sep4)."""
    labels = ["/".join(Path(p).parts[-2:]) for p in tsv_paths]
    if len(set(labels)) < len(labels):
        labels = [str(p) for p in tsv_paths]
    return labels


def model_for_tag(tag: str) -> str | None:
    """Look up the model name for a tag in experiment_log.csv."""
    log = pd.read_csv(EXPERIMENT_LOG)
    matches = log.loc[log["tag"] == tag, "model"]
    return matches.iloc[0] if not matches.empty else None


def runs_for_tag(tag: str) -> list[Path]:
    """Find the archived run logs for a tag: past_runs/agent*/<tag>/results.tsv."""
    paths = sorted(PAST_RUNS_DIR.glob(f"agent*/{tag}/results.tsv"))
    if not paths:
        sys.exit(f"error: no runs found for tag '{tag}' under {PAST_RUNS_DIR}/agent*/<tag>/")
    return paths


def plot_multi_results(
    tsv_paths: list[str | Path],
    out_path: str | Path = "multi_results.png",
    title: str | None = None,
) -> None:
    # One column per run, indexed by experiment idx; runs of different
    # lengths leave NaN at the tail, which the aggregations skip.
    runs = pd.DataFrame(
        {label: last_keep_series(parse_tsv(p)) for label, p in zip(_run_labels(tsv_paths), tsv_paths)}
    )
    # Keep only the experiment range every run covers — past it, the "mean"
    # would be computed from a shrinking subset of runs and read as a trend.
    runs = runs.dropna(axis=0, how="any")

    mean = runs.mean(axis=1)
    lowest = runs.min(axis=1)
    highest = runs.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    # Faint dashed baseline reference line (mean of the runs' baseline keeps).
    baseline_acc = runs.iloc[0].mean()
    ax.axhline(
        y=baseline_acc,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"baseline ({baseline_acc:.3f})",
    )

    # Shaded band between the best and the worst run, then the mean line.
    ax.fill_between(
        mean.index,
        lowest,
        highest,
        color=COLOR_KEEP,
        alpha=0.1,
        linewidth=0,
        label="min–max across runs",
    )
    ax.plot(
        mean.index,
        mean,
        color=COLOR_KEEP,
        linewidth=2,
        solid_capstyle="round",
        label="mean (last keep)",
    )

    ax.set_xlabel("experiment #", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("val acc", fontsize=11, color=INK_SECONDARY)
    ax.set_title(
        title or f"Mean validation accuracy across {len(runs.columns)} runs",
        fontsize=14,
        fontweight="bold",
        color=INK_PRIMARY,
        pad=12,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9, edgecolor=COLOR_GRID)
    _style_axes(ax, len(runs))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


def _sanitize_filename(name: str) -> str:
    """Reduce a model name to filename-safe characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main(argv: list[str] | None = None) -> None:
    """argv: argument list *without* the script name (defaults to sys.argv[1:])."""
    ap = argparse.ArgumentParser(
        description="Plot mean best-keep val_acc progression across several results.tsv runs."
    )
    ap.add_argument("runs", nargs="*", help="results.tsv paths (or omit and pass --tag)")
    ap.add_argument(
        "--tag",
        help="use the archived runs past_runs/agent*/<tag>/results.tsv and look up the "
        "model name for the title in experiment_log.csv",
    )
    ap.add_argument(
        "-o",
        "--output",
        help="output image path (default: Plots/multi/<model>_<tag>.png, else "
        "Plots/multi/multi_results.png)",
    )
    ap.add_argument("--title", help="plot title (default: derived from run count, model and tag)")
    args = ap.parse_args(argv)

    if args.runs:
        paths: list[str | Path] = args.runs
    elif args.tag:
        paths = runs_for_tag(args.tag)
    else:
        ap.error("provide results.tsv paths or --tag")

    model = model_for_tag(args.tag) if args.tag else None
    if args.tag:
        print(f"Tag '{args.tag}': {model or 'unknown model'}, {len(paths)} runs")

    if args.title:
        title = args.title
    elif args.tag:
        title = f"Mean validation accuracy across {len(paths)} runs"
        if model:
            title += f" — {model}"
        title += f" ({args.tag})"
    else:
        title = None

    if args.output:
        out = Path(args.output)
    elif args.tag:
        stem = "_".join(filter(None, [model and _sanitize_filename(model), args.tag]))
        out = PLOTS_DIR / "multi" / f"{stem}.png"
    else:
        out = PLOTS_DIR / "multi" / "multi_results.png"
    plot_multi_results(paths, out, title)


if __name__ == "__main__":
    main(sys.argv[1:])