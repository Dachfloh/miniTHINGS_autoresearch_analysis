#!/usr/bin/env python3
"""Plot train and validation loss across epochs from a run's weight file.

Each fold's `<...>-fold_0.pt` holds the full pickled `GRU_Decoder`, whose
`train_model` loop records per-epoch losses as attributes (`loss`, `val_losses`,
`val_accs`). This script loads them back and plots the two loss curves per
fold, with a marker at the best-epoch (early-stop) point.

Needs an environment with torch and ephyslib importable (the experiment repo's
venv) — the model is pickled by class reference, not by value.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot_results import (
    COLOR_GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PLOTS_DIR,
    _style_axes,
)

# Categorical slots 1/2 of the validated palette (blue/orange).
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"


def load_model(pt_path: str | Path):
    import torch

    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(pt_path, map_location="cpu")
    except (AttributeError, ModuleNotFoundError) as e:
        sys.exit(
            f"error: could not unpickle {pt_path} ({e}). The model is saved by "
            "class reference, so ephyslib must be importable — run this from "
            "an environment with `pip install -e ./packages/ephyslib`."
        )


def epoch_losses(model) -> tuple[list[int], list[float], list[float], float, int]:
    """Extract per-epoch train/val losses from a trained GRU_Decoder.

    The training arrays are preallocated for n_epochs and only filled up to
    the early-stopping point, so the actual epoch count is recovered from the
    filled val_losses rows. Per-epoch train loss is the mean over that epoch's
    batches (matching train_model's own epoch print), and the val loss/acc are
    taken at the last time step or averaged depending on the run's config.
    """
    for attr in ("val_losses", "loss", "val_accs"):
        if not hasattr(model, attr):
            sys.exit(f"error: model has no '{attr}' attribute — the training "
                     "loop in rnn.py may have been modified for this run")
    val_losses = model.val_losses
    train_loss = model.loss
    val_accs = model.val_accs

    filled = (val_losses != 0).any(dim=1)
    n_epochs = int(filled.sum())
    if n_epochs == 0:
        sys.exit("error: model has no recorded epochs (val_losses is empty)")
    # loss is preallocated (nbatch * n_epochs), val_losses (n_epochs, ...):
    # recover the batch count from the shapes, then truncate to the epochs
    # that actually ran (early stopping leaves the tail zeroed)
    nbatch = train_loss.shape[0] // val_losses.shape[0]
    train_loss = train_loss[: n_epochs * nbatch]

    at_last = getattr(model, "loss_at_last_timestep", True)

    def epoch_val(vals):
        return vals[:n_epochs, -1] if at_last else vals[:n_epochs].mean(dim=1)

    val_loss_ep = epoch_val(val_losses)
    val_acc_ep = epoch_val(val_accs)
    train_loss_ep = train_loss.reshape(n_epochs, nbatch).mean(dim=1)

    best_idx = int(val_acc_ep.argmax())  # first max — early stopping keeps the first
    epochs = list(range(1, n_epochs + 1))
    return (
        epochs,
        [float(v) for v in train_loss_ep],
        [float(v) for v in val_loss_ep],
        float(val_acc_ep[best_idx]),
        best_idx + 1,
    )


def plot_loss_curves(run_path: str | Path, out_path: str | Path | None = None) -> None:
    run = Path(run_path)
    if run.is_dir():
        pt_files = sorted(run.glob("*-fold_*.pt"))
        if not pt_files:
            sys.exit(f"error: no *-fold_*.pt weight files in {run}")
        run_name = run.name
    else:
        pt_files = [run]
        run_name = run.parent.name

    fig, axes = plt.subplots(
        len(pt_files), 1, figsize=(10, 6 if len(pt_files) == 1 else 4 + 3 * len(pt_files)),
        layout="constrained", squeeze=False,
    )

    for fold_idx, (ax, pt) in enumerate(zip(axes[:, 0], pt_files)):
        model = load_model(pt)
        epochs, train_loss, val_loss, best_acc, best_ep = epoch_losses(model)

        suffix = f" (fold {fold_idx})" if len(pt_files) > 1 else ""
        ax.plot(
            epochs, train_loss, color=COLOR_TRAIN, linewidth=2,
            label=f"train loss{suffix}", zorder=2,
        )
        ax.plot(
            epochs, val_loss, color=COLOR_VAL, linewidth=2, linestyle="--",
            label=f"val loss{suffix}", zorder=2,
        )
        ax.axvline(best_ep, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)
        ax.annotate(
            f"best val acc {best_acc:.4f} (ep {best_ep})",
            xy=(best_ep, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, -12), textcoords="offset points",
            fontsize=8, color=INK_SECONDARY, va="top",
        )
        ax.set_xlabel("epoch", fontsize=11, color=INK_SECONDARY)
        ax.set_ylabel("loss", fontsize=11, color=INK_SECONDARY)
        if len(pt_files) > 1:
            ax.set_title(f"fold {fold_idx}", fontsize=11, color=INK_PRIMARY, pad=6)
        _style_axes(ax, len(epochs))
        step = max(1, (len(epochs) - 1) // 10)
        ax.set_xticks(list(range(1, len(epochs) + 1, step)))

    fig.suptitle(
        f"Training vs validation loss — {run_name}",
        fontsize=14, fontweight="bold", color=INK_PRIMARY,
    )
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=9, edgecolor=COLOR_GRID)

    if out_path is None:
        out_path = "loss_curves" + "/" + f"{run_name}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run",
        help="run directory (results/rnn_decoding/<run_name>/) or a *-fold_*.pt file",
    )
    parser.add_argument("--out", help="output png path (default: Plots/loss_curves/)")
    args = parser.parse_args()
    plot_loss_curves(args.run, args.out)
