import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import ollama
from sklearn.manifold import TSNE


# One distinct color per repo trajectory; extended only if more repos are
# passed than the palette covers.
REPO_COLORS = [
  '#2a9d8f',  # teal
  '#e76f51',  # coral
  '#264653',  # dark slate
  '#e9c46a',  # mustard
  '#8d99ae',  # slate-blue
  '#9d4edd',  # purple
  '#588157',  # green
  '#d62828',  # red
]


def embed_program(repo_path : str,
                  model : str = 'qwen3-embedding:8b',
                  paths : list[str] = [
                        'allMUA_decoding_rnn.py',
                        'config/rnn_decoder_config.yaml',
                        'packages/ephyslib/ephyslib/decoding/rnn.py'
                        ]):
  """
  Read the program files at `paths` (relative to `repo_path`, the directory
  containing the repo's .git), concatenate them, and embed the result with the
  given ollama embedding model. Returns the embedding vector.
  """

  # Read program

  program = ''

  for path in paths:
    with open(os.path.join(repo_path, path), 'r', encoding='utf-8') as file:
      content_allmua = file.read()

    program += '-----   ' + path + '   -----' + '\n\n' + content_allmua + '\n\n\n'

  # Embed program

  response = ollama.embed(model=model, input=program)
  embeddings = response['embeddings']

  return embeddings[0]


def collect_kept_commits(results_path : str = 'results.tsv') -> list[str]:
  """
  Parse a results.tsv run log and return the commits whose status is 'keep',
  in the order they appear in the file (chronological program trajectory).
  """

  kept = []

  with open(results_path, 'r', encoding='utf-8') as file:
    for line in file:
      line = line.rstrip('\n')

      # Skip the header row and any blank lines.
      if not line or line.startswith('commit'):
        continue

      parts = line.split(None, 3)
      if len(parts) < 3:
        continue

      commit, _, status = parts[0], parts[1], parts[2]

      if status == 'keep':
        kept.append(
          commit
        )

  return kept



def embed_trajectory(repo_path : str,
                     results_path : str = 'results.tsv',
                     model : str = 'qwen3-embedding:8b',
                     paths : list[str] | None = None) -> list[list[float]]:
  """
  Check out each kept commit one by one, embed the program at that commit with
  embed_program, and return the embeddings in chronological order.

  All git operations run inside `repo_path` (the directory containing the
  repo's .git). `results_path`, if relative, is resolved against `repo_path`.

  The original branch/commit is restored afterwards, even if a step fails.
  """
  
  if not os.path.isabs(results_path):
    results_path = os.path.join(repo_path, results_path)

  commits = collect_kept_commits(results_path)

  # Remember where we are so we can restore the working tree afterwards.
  original = subprocess.run(
    ['git', '-C', repo_path, 'symbolic-ref', '--quiet', '--short', 'HEAD'],
    capture_output=True, text=True
  ).stdout.strip()
  if not original:  # detached HEAD -> record the commit hash instead
    original = subprocess.run(
      ['git', '-C', repo_path, 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()

  embeddings = []
  try:
    for i, commit in enumerate(commits):
      print(f'[{i + 1}/{len(commits)}] git -C {repo_path} checkout {commit}')
      subprocess.run(['git', '-C', repo_path, 'checkout', commit], check=True)
      if paths is None:
        embeddings.append(embed_program(repo_path, model=model))
      else:
        embeddings.append(embed_program(repo_path, model=model, paths=paths))
  finally:
    subprocess.run(['git', '-C', repo_path, 'checkout', original], check=True)

  return embeddings



def embed_trajectories(repo_paths : list[str],
                       results_path : str = 'results.tsv',
                       model : str = 'qwen3-embedding:8b',
                       paths : list[str] | None = None) -> list[dict]:
  """
  Embed the program trajectory for each repo in `repo_paths` by calling
  embed_trajectory on each. Returns a list of dicts, one per repo and in the
  same order as the input:

    { 'repo': <label>, 'embeddings': list[list[float]] }

  where 'repo' is the repo directory's basename and 'embeddings' are the
  per-commit embedding vectors in chronological order.
  """

  trajectories = []
  for repo_path in repo_paths:
    label = os.path.basename(os.path.normpath(repo_path))
    print(f'\n=== embedding trajectory: {label} ({repo_path}) ===')
    embeddings = embed_trajectory(
      repo_path, results_path=results_path, model=model, paths=paths
    )
    trajectories.append({'repo': label, 'embeddings': embeddings})
  return trajectories



def read_repo_list(list_path : str) -> list[str]:
  """
  Read repo paths from a file, one per line. Blank lines and lines starting
  with '#' are ignored. Paths are returned verbatim (not resolved), in file
  order.
  """

  repos = []
  with open(list_path, 'r', encoding='utf-8') as file:
    for line in file:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      repos.append(line)
  return repos



def project_trajectories(trajectories : list[dict],
                         perplexity : float | None = None,
                         random_state : int = 0) -> list[dict]:
  """
  Jointly project every repo's embeddings to 2D with a single t-SNE fit so
  that all trajectories share one coordinate space. Adds an 'xy' field
  (np.ndarray of shape (N, 2)) to each trajectory dict and returns the new
  list (the input dicts are not mutated).
  """

  if not trajectories:
    return trajectories

  X = np.vstack([np.asarray(t['embeddings'], dtype=float) for t in trajectories])
  n = X.shape[0]

  if perplexity is None:
    perplexity = min(30.0, max(5.0, n / 3.0))
  # sklearn requires perplexity < n_samples.
  perplexity = min(perplexity, max(1.0, n - 1))

  tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=random_state,
    init='pca',
    learning_rate='auto',
  )
  Y = tsne.fit_transform(X)

  out = []
  i = 0
  for t in trajectories:
    m = len(t['embeddings'])
    tt = dict(t)
    tt['xy'] = Y[i:i + m]
    out.append(tt)
    i += m
  return out



def plot_trajectories(trajectories : list[dict],
                      out_path : str | None = None,
                      title : str = 'Program trajectory (t-SNE of embeddings)'):
  """
  Plot each repo's 2D trajectory as a directed path: a connecting line with an
  arrow between consecutive commits showing the direction of time, a star at
  the first kept commit (start), and a hollow marker at the latest kept commit.
  One color per repo.
  """

  fig, ax = plt.subplots(figsize=(9, 8))

  for idx, t in enumerate(trajectories):
    color = REPO_COLORS[idx % len(REPO_COLORS)]
    xy = t['xy']
    n = xy.shape[0]
    label = t['repo']

    # Connecting path through the commits in chronological order.
    ax.plot(xy[:, 0], xy[:, 1], '-', color=color, alpha=0.45, zorder=1)

    # Direction arrows between consecutive commits.
    for j in range(n - 1):
      ax.annotate(
        '',
        xy=xy[j + 1], xytext=xy[j],
        arrowprops=dict(arrowstyle='-|>', color=color, alpha=0.7, lw=1.2),
        zorder=2,
      )

    # Interior points (skip the start and end, which get special markers).
    if n > 2:
      ax.scatter(xy[1:-1, 0], xy[1:-1, 1], s=40, color=color, alpha=0.85,
                 edgecolors='white', linewidths=0.5, zorder=3)
    ax.scatter(xy[0, 0], xy[0, 1], s=140, color=color, marker='*',
               edgecolors='black', linewidths=0.6, zorder=4,
               label=f'{label} (start)')
    ax.scatter(xy[-1, 0], xy[-1, 1], s=120, color='white', marker='o',
               edgecolors=color, linewidths=2.0, zorder=4,
               label=f'{label} (latest keep)')

  ax.set_title(title)
  ax.set_xlabel('t-SNE 1')
  ax.set_ylabel('t-SNE 2')
  ax.legend(loc='best', fontsize=8, framealpha=0.9)
  ax.set_aspect('equal', adjustable='datalim')
  fig.tight_layout()

  if out_path is not None:
    fig.savefig(out_path, dpi=150)
    print(f'saved figure -> {out_path}')
  plt.show()
  return fig, ax



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Embed program trajectories from multiple git repos (paths '
                'listed in a file), project all embeddings to 2D with t-SNE, '
                'and plot the trajectories.',
  )
  parser.add_argument('repos_file',
                      help='text file of repo paths, one per line '
                           '(blank lines and # comments ignored)')
  parser.add_argument('--results', default='results.tsv',
                      help='results file name inside each repo (default: results.tsv)')
  parser.add_argument('--model', default='qwen3-embedding:8b',
                      help='ollama embedding model (default: qwen3-embedding:8b)')
  parser.add_argument('--paths', nargs='+', default=None,
                      help='program files, relative to each repo, to embed '
                           '(default: the standard miniTHINGS set)')
  parser.add_argument('--perplexity', type=float, default=None,
                      help='t-SNE perplexity (default: min(30, n/3))')
  parser.add_argument('--out', default=None,
                      help='optional output image path (e.g. trajectories.png)')
  args = parser.parse_args()

  repo_paths = read_repo_list(args.repos_file)
  if not repo_paths:
    raise SystemExit(f'no repo paths found in {args.repos_file!r}')

  trajectories = embed_trajectories(
    repo_paths, results_path=args.results, model=args.model, paths=args.paths
  )
  if not trajectories:
    print('no trajectories to plot; nothing to do.')
  else:
    trajectories = project_trajectories(trajectories, perplexity=args.perplexity)
    plot_trajectories(trajectories, out_path=args.out)
