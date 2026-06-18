# `exp/cgasb`: Models from "Classic GNNs are Strong Baselines"

GNN baselines adopted from [Classic GNNs are Strong Baselines](https://arxiv.org/abs/2406.08993).
Training and Optuna hyperparameter tuning for GCN, GAT, GraphSAGE, and LGT.
Reports include predictive performance metrics and efficiency benchmarking.

## How to run

```bash
uv run bin/go.py exp/cgasb/gcn/tolokers-2/tuning.toml --force
```
