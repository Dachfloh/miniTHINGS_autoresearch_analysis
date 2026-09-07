#!/usr/bin/env python3
"""Plot val_acc progression from results.tsv, highlighting keep commits."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    print("pandas is required. Install it with: pip install pandas")
    sys.exit(1)


# Color scheme: keep is a clear teal, discard a muted coral (both validated
# for CVD separation and contrast on a white surface). Baseline and chrome
# stay in neutral grays.
COLOR_KEEP = "#2a9d8f"
COLOR_DISCARD = "#d76f50"
COLOR_BASELINE = "#6b7280"
COLOR_GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def parse_tsv(tsv_path: str | Path, drop_final: bool = True) -> pd.DataFrame:
    """Parse the space-aligned results file.

    The file is space-aligned, not actually tab-delimited, and the
    `description` field is free text containing spaces — so split off the
    first three whitespace-delimited fields and keep the rest verbatim.
    Rows with status "final" (the test-set evaluation of the best config)
    are dropped unless drop_final=False: they are not experiments.
    """
    records = []
    with open(tsv_path) as f:
        header = next(f).split()
        for line in f:
            if not line.strip():
                continue
            try:
                commit, val_acc, status, description = line.rstrip("\n").split(None, 3)
                val_acc = float(val_acc)
            except ValueError:
                # e.g. a description wrapped onto its own line in the source
                print(f"warning: skipping malformed line in {tsv_path}: {line.strip()!r}", file=sys.stderr)
                continue
            if status == "final" and drop_final:
                continue
            records.append([commit, val_acc, status, description])
    df = pd.DataFrame(records, columns=header)
    df["idx"] = range(len(df))
    df["exp_num"] = df["idx"] + 1
    return df


def _style_axes(ax, n_experiments: int) -> None:
    """Shared axis chrome: hairline grid, recessive spines, thinned x ticks."""
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="both", length=3, colors=INK_MUTED)
    # Show every Nth experiment number on the x-axis to avoid crowding.
    tick_step = max(1, n_experiments // 32)
    tick_idx = pd.RangeIndex(0, n_experiments, tick_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(i + 1) for i in tick_idx], fontsize=7.5, rotation=45, ha="right")
    ax.set_xlim(-0.5, n_experiments - 0.5)


PLOTS_DIR = Path(__file__).resolve().parent / "Plots"


def _default_out_path(tsv_path: str | Path) -> Path:
    """Default output for a run plot: the tsv's past_runs path mirrored into Plots/.

    past_runs/agent1/sep4/results.tsv -> Plots/agent1/sep4/results.png
    (variant tsvs like results_nochansel.tsv keep their stem).
    TSVs anywhere else keep the old next-to-the-tsv behavior.
    """
    tsv = Path(tsv_path)
    parts = tsv.resolve().parts
    if "past_runs" in parts:
        return PLOTS_DIR.joinpath(*parts[parts.index("past_runs") + 1 :]).with_suffix(".png")
    return tsv.with_suffix(".png")


def plot_results(tsv_path: str | Path = "results.tsv") -> None:
    df = parse_tsv(tsv_path)

    keep = df[df["status"] == "keep"]
    discard = df[df["status"] == "discard"]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    # Faint dashed baseline reference line (first keep = baseline config).
    baseline_acc = df["val_acc"].iloc[0]
    ax.axhline(
        y=baseline_acc,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"baseline ({baseline_acc:.3f})",
    )

    # Line through the keep commits.
    ax.plot(
        keep["idx"],
        keep["val_acc"],
        color=COLOR_KEEP,
        linewidth=2,
        solid_capstyle="round",
        zorder=2,
    )

    # Scatter points — keeps and discards; crashed experiments have no result
    # and are left out entirely.
    ax.scatter(
        discard["idx"],
        discard["val_acc"],
        color=COLOR_DISCARD,
        s=50,
        zorder=3,
        label="discard",
        edgecolors="white",
        linewidths=1.5,
    )
    ax.scatter(
        keep["idx"],
        keep["val_acc"],
        color=COLOR_KEEP,
        s=70,
        zorder=4,
        label="keep",
        edgecolors="white",
        linewidths=1.5,
    )

    ax.set_xlabel("experiment #", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("val acc", fontsize=11, color=INK_SECONDARY)
    ax.set_title("Validation accuracy progression", fontsize=14, fontweight="bold", color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9, edgecolor=COLOR_GRID)
    _style_axes(ax, len(df))

    out = _default_out_path(tsv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.show()


if __name__ == "__main__":
    plot_results(sys.argv[1] if len(sys.argv) > 1 else "results.tsv")