# miniTHINGS_analysis

Host-side analysis of autoresearch agent runs. This repo is **never** mounted
into the autoresearcher's container — the agent runs in a clone of the
experiment repo (`miniTHINGS_autoresearch`), which has its own `.git` and none
of these analysis files.

## Contents

```
miniTHINGS_analysis/
├── plot_results.py          # Plot val_acc progression from a results.tsv
├── program_trajectory.py    # Embed program files + t-SNE trajectory of kept commits
├── repolist                 # Paths to agent clones (clones of miniTHINGS_autoresearch)
├── past_runs/               # Archived results.tsv / plots from past runs (tracked)
├── plotting/                # Python venv for the analysis scripts (gitignored)
└── t-SNE-trajectories.png   # Example trajectory plot
```

## How the scripts find their targets

Both scripts target the **agent clones** listed in `repolist` (paths like
`…/miniTHINGS_autoresearch-agent1`), not this repo. Those clones are clones of
the experiment repo, so the loop files (`allMUA_decoding_rnn.py`,
`config/rnn_decoder_config.yaml`, `packages/ephyslib/ephyslib/decoding/rnn.py`)
and each run's `results.tsv` live at the clone root.

```bash
# Plot one run's progression:
python plot_results.py past_runs/jul05/results.tsv

# Build a t-SNE trajectory across the agent clones in repolist:
python program_trajectory.py
```

`program_trajectory.py` needs `ollama` (embedding model); `plot_results.py`
needs matplotlib (use the `plotting/` venv).