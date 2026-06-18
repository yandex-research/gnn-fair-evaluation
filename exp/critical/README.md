# `exp/critical`: Models from "A Critical Look at the Evaluation of GNNs under Heterophily"

GNN baselines adopted from [A Critical Look at the Evaluation of GNNs under Heterophily](https://arxiv.org/abs/2302.11640).
Training and Optuna hyperparameter tuning for GCN, GAT, GraphSAGE, and LGT.
Reports include predictive performance metrics and efficiency benchmarking.

## How to run

```bash
uv run bin/go.py exp/critical/gcn/tolokers-2/tuning.toml --force
```
