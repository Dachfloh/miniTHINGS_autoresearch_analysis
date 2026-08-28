# miniTHINGS_analysis

Host-side analysis of autoresearch agent runs. This repo is **never** mounted
into the autoresearcher's container — the agent runs in a clone of the
experiment repo (`miniTHINGS_autoresearch`), which has its own `.git` and none
of these analysis files.

## Contents

```
miniTHINGS_analysis/
├── embed_trajectories.py    # Embed program trajectories (ollama) -> CSV + .npz
├── plot_trajectories.py     # t-SNE + plot trajectories from the bundle
├── plot_results.py          # Plot val_acc progression from a results.tsv
├── repolist                 # TSV list of repos/branches to embed (see below)
├── past_runs/               # Archived results.tsv / plots from past runs (tracked)
├── plotting/                # Python venv for the analysis scripts (gitignored)
└── t-SNE-trajectories.png   # Example trajectory plot
```

## Pipeline: embed, then plot

The analysis is two steps: `embed_trajectories.py` checks out each kept commit,
embeds the program files with ollama, and writes a **bundle** (`trajectories.csv`
+ `trajectories.npz`). `plot_trajectories.py` loads the bundle, t-SNE-projects
the embeddings, and renders a single figure.

### `repolist` — TSV with a header row

The repolist is a TSV file whose first non-comment / non-blank line is a header.
Two columns are required:

- `repo` — path to the agent clone (directory with `.git`).
- `branch` — the autoresearch branch to check out before embedding; may be
  empty to use the current checkout.

Recognized optional column:

- `results` — path to that run's `results.tsv` (relative to the repo or absolute;
  falls back to `--results` when absent).

Any other columns (e.g. `model`) are free-form metadata passed through to the
output bundle and can be used for filtering / coloring in plotting.

Example:

```
# repo\tbranch\tresults\tmodel
/home/.../agent1\tautoresearch/agent1/aug3\tresults.tsv\tglm-5.2
/home/.../agent1\tautoresearch/agent1/aug7\tresults.tsv\tclaude-sonnet-5
```

### Step 1 — embed

```bash
# Basic usage (reads repolist, writes trajectories.csv + trajectories.npz)
python embed_trajectories.py repolist

# Custom output stem, embedding model, and source files to embed
python embed_trajectories.py repolist \
  --out trajectories \
  --model qwen3-embedding:8b \
  --paths allMUA_decoding_rnn.py config/rnn_decoder_config.yaml
```

The script checks out each autoresearch branch, then every kept commit in
`results.tsv`, normalizes the program files (strips comments/docstrings via AST
for Python, round-trips YAML through PyYAML), concatenates them, and embeds with
ollama. The original branch is restored afterwards even on failure.

### Step 2 — plot

```bash
# Activate the plotting venv (has matplotlib, sklearn, pandas, numpy)
source plotting/bin/activate

# --- Six common views ---

# 1. All trajectories, colored individually
python plot_trajectories.py trajectories --color run --out all_by_run.png

# 2. All trajectories, colored by model
python plot_trajectories.py trajectories --color model --out all_by_model.png

# 3. All trajectories, colored by accuracy (val_acc)
python plot_trajectories.py trajectories --color accuracy --out all_by_acc.png

# 4. One model's trajectories, colored individually
python plot_trajectories.py trajectories --filter-model glm-5.2 --out glm_by_run.png

# 5. One model's trajectories, colored by accuracy
python plot_trajectories.py trajectories --filter-model glm-5.2 \
  --color accuracy --out glm_by_acc.png

# 6. One run by itself, colored by accuracy
python plot_trajectories.py trajectories --filter-run agent1:aug3 \
  --out agent1_by_acc.png
```

Color defaults are smart: `model` when showing all runs, `run` when filtering by
model, and `accuracy` when filtering by a single run. `--color` overrides them.

Other useful flags:

- `--perplexity <n>` — t-SNE perplexity (default `min(30, n_points/3)`).
- `--bundle <stem>` — load a bundle other than `trajectories`.
- `--list` — print each trajectory's index, model, and run_label, then exit.
  Use it to look up which indices to pass to `--exclude`.
- `--exclude <idx> [<idx> ...]` — drop trajectories by their 0-based index in
  the catalog (see `--list`). Exclusion applies right after loading, before any
  `--filter-model` / `--filter-run`, so indices always refer to the full catalog.
  Example: `--exclude 4 12`.

```bash
# Find the index of a trajectory you want to drop
python plot_trajectories.py trajectories --list

# Plot everything except trajectories 4 and 12
python plot_trajectories.py trajectories --exclude 4 12 --out selected.png
```

## Other scripts

```bash
# Plot val_acc progression from a single results.tsv
python plot_results.py past_runs/jul05/results.tsv
```

## Dependencies

- `embed_trajectories.py` needs **ollama** (running locally with the embedding
  model, e.g. `qwen3-embedding:8b`).
- `plot_trajectories.py` needs **matplotlib**, **scikit-learn**, **pandas**, and
  **numpy** (install via the `plotting/` venv).