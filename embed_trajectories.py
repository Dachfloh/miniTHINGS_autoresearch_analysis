import argparse
import ast
import csv
import json
import os
import subprocess

import numpy as np
import ollama
import yaml


def _strip_docstrings(tree : ast.AST) -> None:
  """
  Remove docstrings in place: the first statement of any module/function/class
  body that is a bare string-literal expression. A `pass` is left behind where
  stripping would empty a non-module body, so the tree stays unparsable-free.
  """
  for node in ast.walk(tree):
    body = getattr(node, 'body', None)
    if not isinstance(body, list) or not body:
      continue
    first = body[0]
    if (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
      body.pop(0)
      if not body and not isinstance(node, ast.Module):
        body.append(ast.Pass())


def normalize_python(source : str) -> str:
  """
  Standardize Python source for embedding: parse to AST (which drops all
  comments), strip docstrings, then unparse to canonical auto-formatted text.
  Returns the normalized source. The result is not meant to be executed, only
  embedded, so cosmetic differences from the original (e.g. trailing spaces) are
  fine.
  """
  tree = ast.parse(source)
  _strip_docstrings(tree)
  ast.fix_missing_locations(tree)
  return ast.unparse(tree)


def normalize_yaml(source : str) -> str:
  """
  Standardize YAML source for embedding: round-trip through PyYAML, which drops
  comments and normalizes indentation, quoting, and key order (keys sorted).
  """
  data = yaml.safe_load(source)
  if data is None:  # empty / comment-only file
    return ''
  return yaml.safe_dump(
    data, sort_keys=True, default_flow_style=False, allow_unicode=True
  )


def normalize_source(source : str, path : str) -> str:
  """
  Standardize a single program file's text by extension. Python files are
  comment/docstring-stripped and auto-formatted via the AST; YAML files are
  round-tripped through PyYAML. Anything else is returned unchanged. If a file
  cannot be normalized (e.g. syntactically broken code at a mid-edit commit),
  the original text is returned so embedding still proceeds.
  """
  ext = os.path.splitext(path)[1].lower()
  try:
    if ext == '.py':
      return normalize_python(source)
    if ext in ('.yaml', '.yml'):
      return normalize_yaml(source)
  except Exception as exc:
    print(f'  warning: could not normalize {path}: {exc}; using raw text')
  return source


def embed_program(repo_path : str,
                  model : str = 'qwen3-embedding:8b',
                  paths : list[str] = [
                        'allMUA_decoding_rnn.py',
                        'config/rnn_decoder_config.yaml',
                        'packages/ephyslib/ephyslib/decoding/rnn.py'
                        ],
                  normalize : bool = True):
  """
  Read the program files at `paths` (relative to `repo_path`, the directory
  containing the repo's .git), concatenate them, and embed the result with the
  given ollama embedding model. Returns the embedding vector.

  When `normalize` is True (default), each file is standardized before
  concatenation: Python files have comments and docstrings stripped and are
  auto-formatted via the AST, and YAML files are round-tripped through PyYAML.
  This makes the embedding reflect the code's structure rather than
  comment/whitespace churn, so the trajectory plot tracks real changes.
  """

  # Read program

  program = ''

  for path in paths:
    with open(os.path.join(repo_path, path), 'r', encoding='utf-8') as file:
      content_allmua = file.read()

    if normalize:
      content_allmua = normalize_source(content_allmua, path)

    program += '-----   ' + path + '   -----' + '\n\n' + content_allmua + '\n\n\n'

  # Embed program

  response = ollama.embed(model=model, input=program)
  embeddings = response['embeddings']

  return embeddings[0]


def collect_kept_steps(results_path : str = 'results.tsv') -> list[dict]:
  """
  Parse a results.tsv run log and return the kept steps in the order they
  appear (chronological program trajectory). The expected column layout is:

    commit\tval_acc\tstatus\tdescription

  Returns a list of dicts, one per kept step:
    { 'commit': <hash>, 'val_acc': <float or None> }

  'val_acc' is None if the value cannot be parsed as a float (e.g. a crash
  row missing a number), so the step is still embedded.
  """

  steps = []

  with open(results_path, 'r', encoding='utf-8') as file:
    for line in file:
      line = line.rstrip('\n')

      # Skip the header row and any blank lines.
      if not line or line.startswith('commit'):
        continue

      parts = line.split(None, 3)
      if len(parts) < 3:
        continue

      commit, val_acc, status = parts[0], parts[1], parts[2]

      if status == 'keep':
        try:
          val_acc = float(val_acc)
        except ValueError:
          val_acc = None
        steps.append({'commit': commit, 'val_acc': val_acc})

  return steps



def embed_trajectory(repo_path : str,
                     results_path : str = 'results.tsv',
                     model : str = 'qwen3-embedding:8b',
                     paths : list[str] | None = None,
                     normalize : bool = True,
                     branch : str | None = None) -> dict:
  """
  Check out each kept commit one by one, embed the program at that commit with
  embed_program, and return the embeddings together with the per-step accuracy
  (val_acc from results.tsv), both in chronological order.

  When `branch` is given, it is checked out first so that the run's own
  `results.tsv` (which differs per autoresearch branch) is read. When it is
  None, the current checkout is used as-is.

  All git operations run inside `repo_path` (the directory containing the
  repo's .git). `results_path`, if relative, is resolved against `repo_path`.

  The original branch/commit is restored afterwards, even if a step fails.

  Returns:
    { 'embeddings': list[list[float]], 'accuracies': list[float | None] }
  """

  if not os.path.isabs(results_path):
    results_path = os.path.join(repo_path, results_path)

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
  accuracies = []
  try:
    if branch:
      print(f'git -C {repo_path} checkout {branch}')
      subprocess.run(['git', '-C', repo_path, 'checkout', branch], check=True)

    steps = collect_kept_steps(results_path)

    for i, step in enumerate(steps):
      commit = step['commit']
      print(f'[{i + 1}/{len(steps)}] git -C {repo_path} checkout {commit}')
      subprocess.run(['git', '-C', repo_path, 'checkout', commit], check=True)
      if paths is None:
        embeddings.append(embed_program(repo_path, model=model, normalize=normalize))
      else:
        embeddings.append(embed_program(repo_path, model=model, paths=paths, normalize=normalize))
      accuracies.append(step['val_acc'])
  finally:
    subprocess.run(['git', '-C', repo_path, 'checkout', original], check=True)

  return {'embeddings': embeddings, 'accuracies': accuracies}



def embed_trajectories(rows : list[dict],
                       results_path : str = 'results.tsv',
                       model : str = 'qwen3-embedding:8b',
                       paths : list[str] | None = None,
                       normalize : bool = True) -> list[dict]:
  """
  Embed the program trajectory for each input row by calling embed_trajectory
  on it. Each `row` is a dict that must carry a 'repo' key (path to the agent
  clone) and may carry a 'branch' key (the autoresearch branch to check out
  before embedding) and a 'results' key (the results.tsv path for that run,
  relative to the repo or absolute; falls back to `results_path` when absent).
  Any other keys are arbitrary metadata passed through verbatim.

  Returns a list of dicts, one per input row and in the same order, each being
  the input row with added 'embeddings' and 'accuracies' fields (the per-step
  embedding vectors and val_acc values in chronological order):
    { 'repo': <path>, 'branch': <name or ''>, 'results': <path or ''>,
      <...metadata...>, 'embeddings': list[list[float]],
      'accuracies': list[float | None] }
  """

  trajectories = []
  for row in rows:
    repo_path = row['repo']
    branch = row.get('branch') or None
    row_results = row.get('results') or results_path
    label = os.path.basename(os.path.normpath(repo_path))
    branch_repr = f':{branch}' if branch else ''
    print(f'\n=== embedding trajectory: {label}{branch_repr} ({repo_path}) ===')
    result = embed_trajectory(
      repo_path, results_path=row_results, model=model, paths=paths,
      normalize=normalize, branch=branch,
    )
    trajectories.append({**row, **result})
  return trajectories



def read_repolist(list_path : str) -> list[dict]:
  """
  Read a TSV repolist with a header row. The first non-comment, non-blank line
  is the header (tab-separated column names). Each subsequent non-comment,
  non-blank line is a data row mapped to a dict keyed by the header columns.

  The header must contain a 'repo' column (path to the agent clone). A 'branch'
  column is expected (the autoresearch branch to check out; may be empty in a
  row to embed the current checkout). Every other column is free-form metadata
  passed through verbatim.

  Returns a list of row dicts in file order. Short rows are padded with empty
  strings; over-long rows are truncated with a warning.
  """

  rows = []
  with open(list_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

  # Find the header (first non-comment, non-blank line).
  header = None
  header_index = 0
  for i, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
      continue
    header = stripped.split('\t')
    header_index = i
    break
  if header is None:
    raise ValueError(f'no header row found in {list_path!r}')

  for col in ('repo', 'branch'):
    if col not in header:
      raise ValueError(
        f"{list_path!r} header is missing required column {col!r} "
        f"(found: {header!r})"
      )

  ncols = len(header)
  for line in lines[header_index + 1:]:
    stripped = line.rstrip('\n').rstrip('\r')
    if not stripped.strip() or stripped.lstrip().startswith('#'):
      continue
    fields = stripped.split('\t')
    if len(fields) < ncols:
      fields = fields + [''] * (ncols - len(fields))
    elif len(fields) > ncols:
      print(f'  warning: row has {len(fields)} fields but header has '
            f'{ncols}; extra fields ignored: {fields[ncols:]}')
      fields = fields[:ncols]
    rows.append(dict(zip(header, fields)))
  return rows


def write_bundle(trajectories : list[dict],
                 provenance : dict,
                 stem : str) -> tuple[str, str]:
  """
  Write a trajectory bundle to `<stem>.csv` + `<stem>.npz`.

  The CSV is a catalog with one row per trajectory: a leading `id`, every
  metadata column seen on any trajectory (in first-seen order, which mirrors the
  repolist header), then a trailing `n_commits`. A `# meta {...}` comment line
  above the header carries run-level provenance (embedding model, paths, …);
  pandas `read_csv(comment='#')` skips it.

  The .npz holds two arrays per trajectory under `traj_<id>` (the embeddings,
  shape (n_commits, dim), float) and `traj_<id>_acc` (the per-step val_acc,
  shape (n_commits,), float with NaN where val_acc was missing). The catalog
  `id` joins the two.

  Returns (csv_path, npz_path).
  """

  # Catalog columns: 'id', then every metadata key (everything except the
  # per-step 'embeddings'/'accuracies') in first-seen order, then 'n_commits'.
  meta_keys = []
  seen = set()
  for t in trajectories:
    for k in t:
      if k not in ('embeddings', 'accuracies') and k not in seen:
        seen.add(k)
        meta_keys.append(k)
  columns = ['id'] + meta_keys + ['n_commits']

  csv_path = stem + '.csv'
  npz_path = stem + '.npz'

  with open(csv_path, 'w', encoding='utf-8', newline='') as file:
    file.write('# meta ' + json.dumps(provenance) + '\n')
    writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for i, t in enumerate(trajectories):
      row = {'id': i, 'n_commits': len(t['embeddings'])}
      for k in meta_keys:
        row[k] = t.get(k, '')
      writer.writerow(row)

  arrays = {}
  for i, t in enumerate(trajectories):
    arrays[f'traj_{i}'] = np.asarray(t['embeddings'], dtype=float)
    # val_acc may contain None (missing/parse failure) -> NaN in the float array.
    arrays[f'traj_{i}_acc'] = np.asarray(
      [np.nan if a is None else float(a) for a in t['accuracies']], dtype=float
    )
  if arrays:
    np.savez_compressed(npz_path, **arrays)
  else:
    np.savez_compressed(npz_path)  # empty bundle so the file still exists

  return csv_path, npz_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Embed program trajectories from multiple git repos/branches '
                '(listed in a TSV file with a header row) and save them as a '
                'CSV catalog + .npz bundle for later t-SNE projection and '
                'plotting.',
  )
  parser.add_argument('repolist',
                      help='TSV file with a header row. Required columns: '
                           "'repo' (path to the agent clone) and 'branch' "
                           '(the autoresearch branch; may be empty per row). '
                           'Optional recognized columns: results (path to that '
                           "run's results.tsv, relative to the repo or absolute; "
                           'falls back to --results) and any other free-form '
                           'metadata (e.g. model) stored verbatim on each '
                           'trajectory. Blank lines and # comments are ignored.')
  parser.add_argument('--out', default='trajectories',
                      help='output bundle stem: writes <out>.csv (one row per '
                           'trajectory, all metadata + id + n_commits) and '
                           '<out>.npz (one traj_<id> embedding array and one '
                           'traj_<id>_acc accuracy array per id). Any extension '
                           'on --out is replaced (default: trajectories)')
  parser.add_argument('--results', default='results.tsv',
                      help='default results.tsv path for rows without an '
                           'explicit results column; relative paths are '
                           'resolved against each repo (default: results.tsv)')
  parser.add_argument('--model', default='qwen3-embedding:8b',
                      help='ollama embedding model (default: qwen3-embedding:8b)')
  parser.add_argument('--paths', nargs='+', default=None,
                      help='program files, relative to each repo, to embed '
                           '(default: the standard miniTHINGS set)')
  parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                      help='do not strip comments/docstrings or auto-format the '
                           'embedded files before embedding (default: normalize)')
  args = parser.parse_args()

  rows = read_repolist(args.repolist)
  if not rows:
    raise SystemExit(f'no rows found in {args.repolist!r}')

  trajectories = embed_trajectories(
    rows, results_path=args.results, model=args.model, paths=args.paths,
    normalize=args.normalize,
  )

  provenance = {
    'embedding_model': args.model,
    'results': args.results,
    'paths': args.paths,
    'normalize': args.normalize,
  }

  stem = os.path.splitext(args.out)[0]
  csv_path, npz_path = write_bundle(trajectories, provenance, stem)
  print(f'saved catalog    -> {csv_path}')
  print(f'saved embeddings -> {npz_path}')