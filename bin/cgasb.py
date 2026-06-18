import copy
import datetime
import math
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import delu
import numpy as np
import scipy
import scipy.special
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv

import lib
import lib.deep
import lib.graph.data
from lib import KWArgs, PartKey

# >>> Adopted from https://github.com/LUOyk1999/tunedGNN/blob/main/medium_graph/model.py


class _Backbone(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        dropout=0.5,
        heads=1,
        pre_ln=False,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn='gcn',
    ):
        super().__init__()

        self.dropout = dropout
        self.pre_ln = pre_ln

        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)

        if not self.pre_linear:
            if gnn == 'gat':
                self.local_convs.append(
                    GATConv(
                        in_channels,
                        hidden_channels // heads,  # NOTE: had to fix it
                        heads=heads,
                        concat=True,
                        add_self_loops=False,
                        bias=False,
                    )
                )
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            elif gnn == 'gcn':
                self.local_convs.append(
                    GCNConv(
                        in_channels,
                        hidden_channels,
                        cached=False,
                        normalize=True,
                    )
                )
            else:  # NOTE: new conv option
                self.local_convs.append(
                    TransformerConv(
                        in_channels,
                        hidden_channels // heads,
                        heads=heads,
                        concat=True,
                    )
                )
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1

        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(
                    GATConv(
                        hidden_channels,
                        hidden_channels // heads,  # NOTE: had to fix it
                        heads=heads,
                        concat=True,
                        add_self_loops=False,
                        bias=False,
                    )
                )
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif gnn == 'gcn':
                self.local_convs.append(
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                        cached=False,
                        normalize=True,
                    )
                )
            else:  # NOTE: new conv option
                self.local_convs.append(
                    TransformerConv(
                        hidden_channels,
                        hidden_channels // heads,
                        heads=heads,
                        concat=True,
                    )
                )
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self, x, edge_index):
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        for i, local_conv in enumerate(self.local_convs):
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        return x


# <<<


class Model(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        n_classes: None | int,
        backbone: KWArgs,
    ):
        super().__init__()

        backbone = copy.deepcopy(backbone)
        if 'log2_n_heads' in backbone:
            backbone['n_heads'] = 2 ** backbone.pop('log2_n_heads')

        # >>> Backbone
        d_in = n_features
        d_out = 1 if n_classes is None else n_classes

        if 'norm' in backbone:
            assert 'ln' not in backbone and 'bn' not in backbone, (
                "Use either 'norm' or 'ln' and 'bn' flags, not both."
            )
            norm = backbone['norm']
            assert norm in ('none', 'ln', 'bn'), (
                f"backbone['norm'] must be one of 'none', 'ln', 'bn'; got {norm!r}"
            )
            ln = norm == 'ln'
            bn = norm == 'bn'
        else:
            ln = backbone.get('ln', False)
            bn = backbone.get('bn', False)
            assert not (ln and bn), "'ln' and 'bn' are mutually exclusive."

        kwargs = {
            'in_channels': d_in,
            'hidden_channels': backbone['d_hidden'],
            'out_channels': d_out,
            'local_layers': backbone['n_blocks'],
            'dropout': backbone['dropout'],
            'heads': backbone.get('n_heads', 1),
            'pre_linear': backbone['pre_linear'],
            'res': backbone['res'],
            'ln': ln,
            'bn': bn,
            'gnn': backbone['conv'],
        }
        self.backbone = _Backbone(**kwargs)

    def forward(self, edge_index: Tensor, features: dict[str, None | Tensor]) -> Tensor:
        x = []
        for value in features.values():
            if value is None:
                continue
            x.append(value)

        x = torch.cat(x, dim=1)
        x = self.backbone(x, edge_index)
        return x


class Config(TypedDict):
    seed: int
    data: KWArgs
    transform: lib.graph.data.TransformConfig
    model: KWArgs
    optimizer: KWArgs
    n_steps: int
    patience: int
    amp_dtype: NotRequired[Literal['bfloat16', 'float16']]
    save_checkpoint: bool


@lib.catch_oom_exception()
def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
) -> None | lib.JSONDict:
    # >>> Start
    config, output = lib.check(config, output, config_type=Config)
    if not lib.start(main, output, force=force):
        return None

    lib.print_config(config)  # type: ignore
    print()
    delu.random.seed(config['seed'])
    device = lib.get_device()
    logger.info(f'Device: {device}')
    report = lib.create_report(main, config)  # type: ignore

    # >>> Data
    dataset = lib.graph.data.GraphDataset.from_dir(**config['data'])
    assert dataset.task.is_transductive
    features = lib.graph.data.transform_features(
        {key: dataset.data[key] for key in lib.graph.data.GRAPH_FEATURE_KEYS},
        task=dataset.task,
        seed=config['transform']['seed'],
        **config['transform'].get('features', {}),
    )
    dataset.data.update(features)  # type: ignore

    if dataset.task.is_regression and config['transform']['labels']:
        dataset.data['labels'], regression_label_stats = (
            lib.graph.data.standardize_labels(
                dataset.data['labels'], dataset.data['masks']
            )
        )
    else:
        regression_label_stats = None

    dataset = dataset.to_torch(device)
    dataset.data['labels'] = (
        dataset.data['labels'].to(torch.long)  # type: ignore
        if dataset.task.is_classification
        else dataset.data['labels'].to(torch.float)  # type: ignore
    )

    # >>> Model
    model = Model(
        n_features=dataset.n_features,
        n_classes=dataset.task.try_compute_n_classes(),
        **config['model'],
    )
    print(model)

    report['n_parameters'] = lib.deep.get_n_parameters(model)
    logger.info(f'Number of parameters: {report["n_parameters"]}')
    report['prediction_type'] = 'labels' if dataset.task.is_regression else 'probs'
    model.to(device)

    # >>> Train
    optimizer = lib.deep.make_optimizer(
        **config['optimizer'], params=lib.deep.make_parameter_groups(model)
    )
    base_loss_fn = (
        nn.functional.mse_loss
        if dataset.task.is_regression
        else nn.functional.cross_entropy
    )

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return base_loss_fn(y_pred, y_true)

    step = 0
    training_log = []
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(config['patience'], mode='max')

    amp_dtype = config.get('amp_dtype')
    if amp_dtype == 'bfloat16':
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'amp_dtype is set to "bfloat16" in the config.'
                f' However, the current {device.type.upper()} device'
                ' does not support bfloat16'
            )
        amp_dtype = torch.bfloat16

    elif amp_dtype == 'float16':
        amp_dtype = torch.float16

    grad_scaler = (
        torch.amp.GradScaler(device.type)  # type: ignore
        if amp_dtype is torch.float16
        else None
    )
    logger.info(f'AMP dtype: {amp_dtype}')

    graph = dataset.data['graph']
    features = {key: dataset.data[key] for key in lib.graph.data.GRAPH_FEATURE_KEYS}
    edge_index = torch.stack(graph.edges(), dim=0).to(torch.int64)

    @torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    def apply_model() -> Tensor:
        outputs = model(edge_index, features)
        assert outputs.ndim == 2
        return outputs.squeeze(-1).float()

    @torch.inference_mode()
    def evaluate() -> tuple[dict[PartKey, Any], dict[PartKey, np.ndarray]]:
        model.eval()
        outputs = apply_model()
        outputs = outputs.cpu().numpy()

        if dataset.task.is_regression and regression_label_stats is not None:
            _predictions: np.ndarray = (
                outputs * regression_label_stats.std + regression_label_stats.mean
            )
        else:
            _predictions: np.ndarray = scipy.special.softmax(outputs, axis=-1)
            if dataset.task.is_binclass:
                _predictions = _predictions[..., 1]

        predictions: dict[PartKey, np.ndarray] = {
            part: _predictions[dataset.task.masks[part]] for part in lib.DATA_PARTS
        }
        metrics = (
            dataset.task.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else lib.get_default_metrics()
        )
        return metrics, predictions

    def save_checkpoint() -> None:
        lib.dump_checkpoint(
            output,
            {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),  # type: ignore
                'random_state': delu.random.get_state(),
                'early_stopping': early_stopping,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
        )

    print()
    if device.type == 'cuda':
        delu.cuda.free_memory()
        torch.cuda.reset_peak_memory_stats(device)

    timer.run()
    while step < config['n_steps']:
        step_start_time = timer.elapsed()

        model.train()
        optimizer.zero_grad()

        outputs = apply_model()
        targets = dataset.data['labels']

        loss = loss_fn(
            outputs[dataset.data['masks']['train']],
            targets[dataset.data['masks']['train']],
        )
        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        step += 1
        step_end_time = timer.elapsed()
        step_loss = loss.detach().item()

        metrics, predictions = evaluate()
        val_score_improved = (
            'metrics' not in report
            or metrics['val']['score'] > report['metrics']['val']['score']
        )

        training_log.append(
            {
                'metrics': metrics,
                'time': timer.elapsed(),
            }
        )
        print(
            f'{"+" if val_score_improved else " "}'
            f' [step] {step:>4}'
            f' [val] {metrics["val"]["score"]:.3f}'
            f' [test] {metrics["test"]["score"]:.3f}'
            f' [loss] {step_loss:.4f}'
            f' [time] {datetime.timedelta(seconds=math.trunc(timer.elapsed()))}'
            f' [s/it] {(step_end_time - step_start_time):.4f}'
        )

        if val_score_improved:
            report['best_step'] = step
            report['metrics'] = metrics
            lib.dump_report(output, report)
            lib.dump_predictions(output, predictions)
            lib.dump_summary(output, lib.summarize(report))
            if config['save_checkpoint']:
                save_checkpoint()

        early_stopping.update(metrics['val']['score'])
        if not lib.are_valid_predictions(predictions) or early_stopping.should_stop():
            break

    if device.type == 'cuda':
        torch.cuda.synchronize()
        report['training_max_memory_allocated_bytes'] = (  #
            torch.cuda.max_memory_allocated(device)
        )

    report['time'] = timer.elapsed()

    # >>> Benchmark inference

    optimizer = None  # type: ignore
    grad_scaler = None
    if device.type == 'cuda':
        delu.cuda.free_memory()
        torch.cuda.reset_peak_memory_stats(device)

    timer.reset()
    timer.run()
    evaluate()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        report['inference_max_memory_allocated_bytes'] = (
            torch.cuda.max_memory_allocated(device)
        )

    report['inference_time'] = timer.elapsed()

    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.init()
    lib.run(main)
