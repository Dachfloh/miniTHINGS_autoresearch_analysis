#!/usr/bin/env python3
"""Plot val_acc progression from results.tsv, highlighting keep commits."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

try:
    import pandas as pd
except ImportError:
    print("pandas is required. Install it with: pip install pandas")
    sys.exit(1)


# Color scheme: keep is a clear teal/green, discard a muted coral,
# crash a neutral gray, and the connecting line very light.
COLOR_KEEP = "#2a9d8f"
COLOR_DISCARD = "#e07a5f"
COLOR_CRASH = "#9ca3af"
COLOR_LINE = "#9ca3af"
COLOR_BASELINE = "#6b7280"
COLOR_BASELINE_LINE = "#6b7280"
COLOR_BASELINE_DOT = "#f4a261"
COLOR_BEST = "#264653"


def _parse_tsv(tsv_path: str | Path) -> pd.DataFrame:
    """Parse the space-aligned results file.

    The file is space-aligned, not actually tab-delimited, and the
    `description` field is free text containing spaces — so split off the
    first three whitespace-delimited fields and keep the rest verbatim.
    """
    records = []
    with open(tsv_path) as f:
        header = next(f).split()
        for line in f:
            if not line.strip():
                continue
            commit, val_acc, status, description = line.rstrip("\n").split(None, 3)
            records.append([commit, float(val_acc), status, description])
    df = pd.DataFrame(records, columns=header)
    df["idx"] = range(len(df))
    df["exp_num"] = df["idx"] + 1
    return df


def plot_results(tsv_path: str | Path = "results.tsv") -> None:
    df = _parse_tsv(tsv_path)

    keep_mask = df["status"] == "keep"
    discard_mask = df["status"] == "discard"
    crash_mask = df["status"] == "crash"

    keep = df[keep_mask].copy()
    keep["keep_num"] = range(1, len(keep) + 1)
    non_baseline_keep = keep.iloc[1:].copy()

    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    table_ax = fig.add_subplot(gs[1])
    table_ax.axis("off")

    # Baseline reference line and dot.
    baseline_acc = df.loc[0, "val_acc"]
    ax.axhline(
        y=baseline_acc,
        color=COLOR_BASELINE_LINE,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"baseline ({baseline_acc:.3f})",
    )
    ax.scatter(
        [df.loc[0, "idx"]],
        [baseline_acc],
        color=COLOR_BASELINE_DOT,
        s=110,
        zorder=4,
        marker="D",
        edgecolors="white",
        linewidth=1,
    )
    ax.annotate(
        "1",
        (df.loc[0, "idx"], baseline_acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="white",
        zorder=5,
        bbox=dict(
            boxstyle="circle,pad=0.12",
            facecolor=COLOR_BASELINE_DOT,
            edgecolor="white",
            linewidth=0.8,
        ),
    )

    # Connecting line through all plottable (non-crash) experiments.
    plottable = df[~crash_mask]
    ax.plot(
        plottable["idx"],
        plottable["val_acc"],
        color=COLOR_LINE,
        linewidth=1.2,
        zorder=1,
    )

    # Discard points.
    discard = df[discard_mask]
    if not discard.empty:
        ax.scatter(
            discard["idx"],
            discard["val_acc"],
            color=COLOR_DISCARD,
            s=45,
            zorder=2,
            label="discard",
            marker="o",
            edgecolors="white",
            linewidth=0.5,
        )

    # Crash points.
    crash = df[crash_mask]
    if not crash.empty:
        ax.scatter(
            crash["idx"],
            crash["val_acc"],
            color=COLOR_CRASH,
            s=70,
            zorder=2,
            label="crash",
            marker="x",
            linewidth=2,
        )

    # Connecting line through keep commits, including the baseline diamond.
    ax.plot(
        keep["idx"],
        keep["val_acc"],
        color=COLOR_KEEP,
        linewidth=1.8,
        linestyle="-",
        alpha=0.65,
        zorder=3,
    )

    # Keep points — larger, on top, with numbered callouts (excluding the baseline).
    ax.scatter(
        non_baseline_keep["idx"],
        non_baseline_keep["val_acc"],
        color=COLOR_KEEP,
        s=110,
        zorder=4,
        label="keep",
        marker="o",
        edgecolors="white",
        linewidth=1,
    )

    best_idx = non_baseline_keep["val_acc"].idxmax()
    for _, row in non_baseline_keep.iterrows():
        is_best = row.name == best_idx
        ax.annotate(
            f"{row['keep_num']}",
            (row["idx"], row["val_acc"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=5,
            bbox=dict(
                boxstyle="circle,pad=0.12",
                facecolor=COLOR_KEEP,
                edgecolor="white",
                linewidth=0.8,
            ),
        )
        if is_best:
            ax.annotate(
                f"{row['val_acc']:.3f}",
                (row["idx"], row["val_acc"]),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=COLOR_KEEP,
                zorder=5,
            )

    # Axis styling.
    ax.set_xlabel("experiment #", fontsize=11)
    ax.set_ylabel("val acc", fontsize=11)
    ax.set_title("Validation accuracy progression", fontsize=15, fontweight="bold", pad=12)
    # Build legend entries, including a synthetic diamond handle for the baseline.
    baseline_handle = Line2D(
        [],
        [],
        color=COLOR_BASELINE_LINE,
        linestyle="--",
        linewidth=1,
        marker="D",
        markerfacecolor=COLOR_BASELINE_DOT,
        markeredgecolor="white",
        markersize=8,
        label=f"baseline ({baseline_acc:.3f})",
    )
    handles, labels = ax.get_legend_handles_labels()
    # Replace the dashed-line baseline handle with the combined diamond+line one.
    for i, label in enumerate(labels):
        if label.startswith("baseline"):
            handles[i] = baseline_handle
            break
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        edgecolor="#e5e7eb",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")

    # Show every other experiment number on the x-axis to avoid crowding.
    tick_step = max(1, len(df) // 32)
    tick_idx = df["idx"][::tick_step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(i + 1) for i in tick_idx], fontsize=7.5, rotation=45, ha="right")
    ax.tick_params(axis="both", length=3, color="#6b7280")
    ax.set_xlim(-0.5, len(df) - 0.5)

    # Description table for keep commits.
    table_data = [
        [f"{row['keep_num']}", f"{row['val_acc']:.3f}", row["description"]]
        for _, row in keep.iterrows()
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=["#", "val acc", "keep description"],
        loc="upper left",
        colWidths=[0.05, 0.09, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)

    # Style the table header and cells.
    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_edgecolor("#e5e7eb")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(fontweight="bold", color="#374151")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#374151")
        # Left-align the description column, center the first two.
        cell.set_text_props(ha="left" if col == 2 else "center")

    out = Path(tsv_path).with_suffix(".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")
    plt.show()


if __name__ == "__main__":
    plot_results(sys.argv[1] if len(sys.argv) > 1 else "results.tsv")
