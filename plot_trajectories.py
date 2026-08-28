import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


# One distinct color per trajectory / model; extended only if needed.
PALETTE = [
  '#2a9d8f',  # teal
  '#e76f51',  # coral
  '#264653',  # dark slate
  '#e9c46a',  # mustard
  '#8d99ae',  # slate-blue
  '#9d4edd',  # purple
  '#588157',  # green
  '#d62828',  # red
]

ACC_CMAP = 'viridis'
COLORS = ('run', 'model', 'accuracy')

# Fixed color for every trajectory's baseline (start) star, regardless of the
# active color mode, so the start point reads consistently across the figure.
BASELINE_STAR_COLOR = '#ffd60a'


def run_label(repo : str, branch : str) -> str:
  """
  Short label for an individual run, used in legends/titles when coloring by
  run. Autoresearch branches are named `autoresearch/<agent>/<date>`, so we
  collapse to `<agent>:<date>` (e.g. agent1:aug3); any other non-empty branch is
  used verbatim; a missing branch falls back to the repo's basename.
  """
  if branch:
    parts = branch.split('/')
    if len(parts) >= 3 and parts[0] == 'autoresearch':
      return f'{parts[-2]}:{parts[-1]}'
    return branch
  return os.path.basename(os.path.normpath(repo))


def load_catalog(stem : str) -> tuple[list[dict], dict]:
  """
  Load only the `<stem>.csv` catalog (no embeddings). Returns (trajectories,
  provenance). Each trajectory dict carries the catalog columns plus a derived
  `run_label`, but no `embeddings`/`accuracies` — used by `--list`, which only
  needs the text columns.
  """
  csv_path = stem + '.csv'

  provenance = {}
  with open(csv_path, 'r', encoding='utf-8') as file:
    first = file.readline()
    if first.startswith('# meta '):
      provenance = json.loads(first[len('# meta '):])

  catalog = pd.read_csv(csv_path, comment='#', keep_default_na=False)

  trajectories = []
  for _, row in catalog.iterrows():
    tid = int(row['id'])
    t = row.to_dict()
    t['id'] = tid
    t['run_label'] = run_label(t['repo'], t.get('branch', ''))
    trajectories.append(t)
  return trajectories, provenance


def load_bundle(stem : str) -> tuple[list[dict], dict]:
  """
  Load the `<stem>.csv` catalog + `<stem>.npz` embeddings written by
  embed_trajectories.py. Returns (trajectories, provenance).

  Each trajectory dict carries every catalog column (id, repo, branch, model,
  results, ...), the `embeddings` and `accuracies` arrays from the .npz, and a
  derived `run_label`. `provenance` is the run-level metadata from the catalog's
  `# meta` comment line (embedding model, paths, ...).
  """
  trajectories, provenance = load_catalog(stem)
  npz = np.load(stem + '.npz')
  for t in trajectories:
    tid = t['id']
    t['embeddings'] = npz[f'traj_{tid}']
    t['accuracies'] = npz[f'traj_{tid}_acc']
  return trajectories, provenance


def fit_tsne(trajs : list[dict],
             perplexity : float | None = None,
             random_state : int = 0) -> None:
  """
  Jointly t-SNE-project the embeddings of every trajectory in `trajs` to 2D so
  they share one coordinate space, assigning an `xy` array (shape (n, 2)) to
  each trajectory dict in place. A group with fewer than 2 total points gets
  all-zero coordinates (t-SNE is undefined there).
  """
  if not trajs:
    return
  X = np.vstack([np.asarray(t['embeddings'], dtype=float) for t in trajs])
  n = X.shape[0]
  if n < 2:
    for t in trajs:
      t['xy'] = np.zeros((len(t['embeddings']), 2))
    return

  if perplexity is None:
    perplexity = min(30.0, max(5.0, n / 3.0))
  perplexity = min(perplexity, max(1.0, n - 1))

  tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=random_state,
    init='pca',
    learning_rate='auto',
  )
  Y = tsne.fit_transform(X)

  i = 0
  for t in trajs:
    m = len(t['embeddings'])
    t['xy'] = Y[i:i + m]
    i += m


def _categorical_color(item : str, palette_idx : int) -> str:
  return PALETTE[palette_idx % len(PALETTE)]


def draw_trajectory(ax, t : dict, color : str,
                    ctx : SimpleNamespace, idx : int) -> str:
  """
  Draw a single trajectory on `ax`. Returns the label string (used for legend
  deduplication).
  """
  xy = t['xy']
  n = xy.shape[0]

  if color == 'accuracy':
    acc = np.ma.masked_invalid(t['accuracies'])
    rgba = ctx.acc_cmap(ctx.acc_norm(acc))

    ax.plot(xy[:, 0], xy[:, 1], '-', color='#888888', alpha=0.45, zorder=1)
    for j in range(n - 1):
      ax.annotate(
        '', xy=xy[j + 1], xytext=xy[j],
        arrowprops=dict(arrowstyle='-|>', color='#888888', alpha=0.6, lw=1.0),
        zorder=2,
      )
    if n > 2:
      ax.scatter(xy[1:-1, 0], xy[1:-1, 1], s=40, c=rgba[1:-1],
                 alpha=0.9, edgecolors='white', linewidths=0.5, zorder=3)
    ax.scatter(xy[0, 0], xy[0, 1], s=160, color=BASELINE_STAR_COLOR,
               marker='*', edgecolors='black', linewidths=0.6, zorder=4)
    if n > 1:
      ax.scatter(xy[-1, 0], xy[-1, 1], s=110, color=rgba[-1], marker='o',
                 edgecolors='black', linewidths=1.5, zorder=4)
    return t['run_label']

  if color == 'run':
    cc = _categorical_color(t['run_label'], idx)
    label = t['run_label']
  else:  # color == 'model'
    cc = ctx.model_colors[t.get('model', '')]
    label = t.get('model', '')

  ax.plot(xy[:, 0], xy[:, 1], '-', color=cc, alpha=0.45, zorder=1)
  for j in range(n - 1):
    ax.annotate(
      '', xy=xy[j + 1], xytext=xy[j],
      arrowprops=dict(arrowstyle='-|>', color=cc, alpha=0.7, lw=1.2),
      zorder=2,
    )
  if n > 2:
    ax.scatter(xy[1:-1, 0], xy[1:-1, 1], s=40, color=cc, alpha=0.85,
               edgecolors='white', linewidths=0.5, zorder=3)
  ax.scatter(xy[0, 0], xy[0, 1], s=140, color=BASELINE_STAR_COLOR, marker='*',
             edgecolors='black', linewidths=0.6, zorder=4)
  if n > 1:
    ax.scatter(xy[-1, 0], xy[-1, 1], s=120, color='white', marker='o',
               edgecolors=cc, linewidths=2.0, zorder=4)
  return label


def build_single_figure(trajectories : list[dict],
                        color : str,
                        perplexity : float | None,
                        filter_model : str | None,
                        filter_run : str | None) -> 'plt.Figure':
  """
  Build a single figure: filter trajectories by model/run if requested, fit
  one joint t-SNE on the subset, and draw everything on one axis with the chosen
  color mode.
  """
  filtered = trajectories
  title_parts : list[str] = []

  if filter_run:
    filtered = [t for t in filtered if t['run_label'] == filter_run]
    if not filtered:
      raise SystemExit(f'no trajectory matched run_label {filter_run!r}')
    title_parts.append(f'Run: {filter_run}')
  elif filter_model:
    filtered = [t for t in filtered if t.get('model', '') == filter_model]
    if not filtered:
      raise SystemExit(f'no trajectory matched model {filter_model!r}')
    title_parts.append(f'Model: {filter_model}')
  else:
    title_parts.append('All trajectories')

  fit_tsne(filtered, perplexity=perplexity)

  # Build color context.
  models = sorted({t.get('model', '') for t in filtered})
  model_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

  acc_cmap = plt.get_cmap(ACC_CMAP).with_extremes(bad='lightgray')
  all_acc = np.concatenate([t['accuracies'] for t in filtered]) \
      if filtered else np.array([])
  valid = all_acc[~np.isnan(all_acc)] if all_acc.size else np.array([])
  acc_norm = Normalize(
    vmin=float(valid.min()) if valid.size else 0.0,
    vmax=float(valid.max()) if valid.size else 1.0,
  )
  ctx = SimpleNamespace(model_colors=model_colors, acc_cmap=acc_cmap,
                        acc_norm=acc_norm)

  fig, ax = plt.subplots(figsize=(9, 8))

  legend_items : list[tuple[str, str]] = []
  for idx, t in enumerate(filtered):
    label = draw_trajectory(ax, t, color, ctx, idx)
    if color != 'accuracy':
      legend_items.append((label, ctx.model_colors.get(label)
                           if color == 'model' else _categorical_color(label, idx)))

  if color != 'accuracy' and legend_items:
    seen : dict[str, str] = {}
    for label, cc in legend_items:
      seen.setdefault(label, cc)
    if len(seen) > 1:
      handles = [
        Line2D([0], [0], color=cc, lw=2, marker='o', markersize=6, label=label)
        for label, cc in seen.items()
      ]
      ax.legend(handles=handles, loc='best', fontsize=8, framealpha=0.9)

  ax.set_xlabel('t-SNE 1')
  ax.set_ylabel('t-SNE 2')
  ax.set_aspect('equal', adjustable='datalim')
  title_parts.append(f'colored by {color}')
  ax.set_title('  '.join(title_parts), fontsize=11)

  if color == 'accuracy':
    # Reserve space on the right for the colorbar.
    fig.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.07)
    sm = ScalarMappable(norm=acc_norm, cmap=acc_cmap)
    fig.colorbar(sm, ax=ax, label='val_acc', shrink=0.8)
  else:
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
  return fig


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Load a program-trajectory bundle (CSV catalog + .npz, as '
                'produced by embed_trajectories.py), t-SNE-project the '
                'embeddings, and plot trajectories on a single figure.',
  )
  parser.add_argument('bundle', nargs='?', default='trajectories',
                      help='bundle stem: reads <stem>.csv + <stem>.npz '
                           '(default: trajectories)')
  parser.add_argument('--color', choices=COLORS, default=None,
                      help='color points by run, model, or val_acc. '
                           'Default: model (all runs), run (when --filter-model '
                           'is given), accuracy (when --filter-run is given)')
  parser.add_argument('--filter-model', default=None, metavar='MODEL',
                      help='only include trajectories whose model column equals '
                           'MODEL')
  parser.add_argument('--filter-run', default=None, metavar='LABEL',
                      help='only include the trajectory whose run_label '
                           '(agent:date, e.g. agent1:aug3) equals LABEL')
  parser.add_argument('--exclude', nargs='+', type=int, default=None,
                      metavar='IDX',
                      help='exclude trajectories by their 0-based index in the '
                           'catalog (see --list). E.g. --exclude 0 3 7')
  parser.add_argument('--list', action='store_true',
                      help='print each trajectory index with its run_label and '
                           'model, then exit (use to find indices for --exclude)')
  parser.add_argument('--perplexity', type=float, default=None,
                      help='t-SNE perplexity (default: min(30, n/3))')
  parser.add_argument('--out', default=None,
                      help='optional output image path (e.g. trajectories.png)')
  args = parser.parse_args()

  # Default color depends on filters.
  if args.color is None:
    if args.filter_run:
      args.color = 'accuracy'
    elif args.filter_model:
      args.color = 'run'
    else:
      args.color = 'model'

  stem = os.path.splitext(args.bundle)[0]

  if args.list:
    # Only needs the CSV columns; skip the .npz load entirely.
    trajectories, _ = load_catalog(stem)
    for i, t in enumerate(trajectories):
      print(f'{i:3d}  {t.get("model", "") or "(no model)":20s}  '
            f'{t["run_label"]}')
    sys.exit(0)

  trajectories, _provenance = load_bundle(stem)

  if not trajectories:
    print('no trajectories to plot; nothing to do.')
    sys.exit(0)

  if args.exclude:
    excluded = set(args.exclude)
    n = len(trajectories)
    bad = sorted(i for i in excluded if not 0 <= i < n)
    if bad:
      raise SystemExit(f'--exclude index out of range (0..{n - 1}): {bad}')
    trajectories = [t for i, t in enumerate(trajectories) if i not in excluded]
    print(f'excluded {len(excluded)} trajectory/ies by index '
          f'({sorted(excluded)}); {len(trajectories)} remaining.')
    if not trajectories:
      print('nothing left to plot; nothing to do.')
      sys.exit(0)

  fig = build_single_figure(
    trajectories, args.color, args.perplexity,
    filter_model=args.filter_model, filter_run=args.filter_run,
  )
  if args.out is not None:
    fig.savefig(args.out, dpi=150)
    print(f'saved figure -> {args.out}')
  plt.show()