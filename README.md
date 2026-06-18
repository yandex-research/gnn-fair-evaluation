# A Fair Evaluation of Graph Foundation Models for Node Property Prediction

This work has been accepted to the [Workshop on Graph Foundation Models @ ICML 2026](https://openreview.net/group?id=ICML.cc/2026/Workshop/GFM).

> [!NOTE]
> Link to the paper will be added later.

> [!IMPORTANT]
> This repository provides only the code for reproducing the experimental results of GNNs. See [this section](#code-for-graph-foundation-models) for the links to GFMs repositories.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

1. Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync the environment:
```bash
uv sync --managed-python
```

This will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

## Data

We use [GraphLand datasets](https://zenodo.org/records/16895532). After downloading, extract the data and create a symlink to the `data` directory:
```bash
ln -s /path/to/downloaded/data data
```

## Running Experiments

The `bin/go.py` script runs the full experimental pipeline:
1. Hyperparameter tuning (if `tuning.toml` is provided)
2. Generating `evaluation.toml` with the best configuration
3. Evaluating the best configuration across multiple seeds
4. Ensembling predictions from the best models

The script accepts either `tuning.toml` (to run the full pipeline) or `evaluation.toml` (to skip tuning and run evaluation directly).

```bash
uv run bin/go.py exp/<experiment>/<model>/<dataset>/tuning.toml
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | (required) | Path to a `tuning.toml` or `evaluation.toml` config file |
| `--n_seeds` | 5 | Number of seeds for evaluation |
| `--ensemble_size` | 5 | Size of ensemble (set to 0 to skip ensembling) |
| `--continue` | False | Continue an unfinished experiment |
| `--force` | False | Force overwrite existing results |

### Example

```bash
uv run bin/go.py exp/cgasb/gcn/tolokers-2/tuning.toml --force
```

> [!NOTE]
> Since all experiment reports are included in this repository, you must pass `--force` to re-run experiments and overwrite existing results if want to reproduce them.

## Experiment Directories

| Directory | Description |
|:----------|:------------|
| `exp/cgasb` | GNN baselines from [Classic GNNs are Strong Baselines](https://arxiv.org/abs/2406.08993) |
| `exp/critical` | GNN baselines from [A Critical Look at the Evaluation of GNNs under Heterophily](https://arxiv.org/abs/2302.11640) |

## Code for Graph Foundation Models

To reproduce the experimental results of GFMs, use their official repositories:

| Method | GitHub Repository |
|:-------|:-----------|
| AnyGraph | [HKUDS/AnyGraph](https://github.com/HKUDS/AnyGraph) |
| OpenGraph | [HKUDS/OpenGraph](https://github.com/HKUDS/OpenGraph) |
| GCOPE | [cshhzhao/gcope](https://github.com/cshhzhao/gcope) |
| TS-GNN | [benfinkelshtein/EquivarianceEverywhere](https://github.com/benfinkelshtein/EquivarianceEverywhere) |
| MDGFM | [wbkzwqtzw/MDGFM](https://github.com/wbkzwqtzw/MDGFM) |
| SAMGPT | [blue-soda/SAMGPT](https://github.com/blue-soda/SAMGPT) |
| TAG | [ahayler/tag](https://github.com/ahayler/tag) |
| G2T | [yandex-research/G2T-FM](https://github.com/yandex-research/G2T-FM) |
| GraphPFN | [yandex-research/graphpfn](https://github.com/yandex-research/graphpfn) |

## Project Structure

```
.
├── bin/                 # Executable scripts
├── lib/                 # Core library code
├── exp/                 # Experiment configurations and results
├── data/                # Symlink to datasets
└── pyproject.toml       # Project configuration and dependencies
```
