import copy
import datetime
import math
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import delu
import dgl
import numpy as np
import scipy
import scipy.special
import torch
import torch.amp
import torch.nn as nn
from dgl import ops
from dgl.nn.functional import edge_softmax
from loguru import logger
from torch import Tensor

import lib
import lib.deep
import lib.graph.data
from lib import KWArgs, PartKey

# >>> Adopted from https://github.com/yandex-research/heterophilous-graphs


class _ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res
        return x


class _FeedForwardModule(nn.Module):
    def __init__(
        self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs
    ):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class _GCNModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = _FeedForwardModule(
            dim=dim, hidden_dim_multiplier=hidden_dim_multiplier, dropout=dropout
        )

    def forward(self, graph, x):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)  # type: ignore
        norm_coefs = 1 / degree_edge_products**0.5

        x = ops.u_mul_e_sum(graph, x, norm_coefs)  # type: ignore
        x = self.feed_forward_module(graph, x)
        return x


class _SAGEModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super().__init__()
        self.feed_forward_module = _FeedForwardModule(
            dim=dim,
            input_dim_multiplier=2,
            hidden_dim_multiplier=hidden_dim_multiplier,
            dropout=dropout,
        )

    def forward(self, graph, x):
        message = ops.copy_u_mean(graph, x)  # type: ignore
        x = torch.cat([x, message], dim=1)
        x = self.feed_forward_module(graph, x)
        return x


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError(
            'Dimension mismatch: hidden_dim should be a multiple of num_heads.'
        )


class _GATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(
            in_features=dim, out_features=num_heads, bias=False
        )
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = _FeedForwardModule(
            dim=dim, hidden_dim_multiplier=hidden_dim_multiplier, dropout=dropout
        )

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)  # type: ignore
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)  # type: ignore
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)
        return x


class _GATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(
            in_features=dim, out_features=num_heads, bias=False
        )
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = _FeedForwardModule(
            dim=dim,
            input_dim_multiplier=2,
            hidden_dim_multiplier=hidden_dim_multiplier,
            dropout=dropout,
        )

    def forward(self, graph, x):
        x = self.input_linear(x)

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)  # type: ignore
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)  # type: ignore
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], dim=1)

        x = self.feed_forward_module(graph, x)
        return x


class _TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim**0.5  # type: ignore
        attn_probs = edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)  # type: ignore
        x = x.reshape(-1, self.dim)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


class _TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim**0.5  # type: ignore
        attn_probs = edge_softmax(graph, attn_scores)

        message = ops.u_mul_e_sum(graph, values, attn_probs)  # type: ignore
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], dim=1)

        x = self.output_linear(x)
        x = self.dropout(x)
        return x


_MODULES: dict[str, list[type[nn.Module]]] = {
    'GCN': [_GCNModule],
    'SAGE': [_SAGEModule],
    'GAT': [_GATModule],
    'GAT-sep': [_GATSepModule],
    'GT': [_TransformerAttentionModule, _FeedForwardModule],
    'GT-sep': [_TransformerAttentionSepModule, _FeedForwardModule],
}

_NORMALIZATION: dict[str, type[nn.Module]] = {
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d,
    'Identity': nn.Identity,
}


class _Backbone(nn.Module):
    def __init__(
        self,
        model_name,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_dim_multiplier,
        num_heads,
        normalization,
        dropout,
    ):
        super().__init__()

        normalization = _NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in _MODULES[model_name]:
                residual_module = _ResidualModuleWrapper(
                    module=module,
                    normalization=normalization,
                    dim=hidden_dim,
                    hidden_dim_multiplier=hidden_dim_multiplier,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x)  # NOTE: removed squeezing
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
        if 'log2_hidden_dim_multiplier' in backbone:
            backbone['hidden_dim_multiplier'] = 2 ** backbone.pop(
                'log2_hidden_dim_multiplier'
            )

        conv = backbone['conv']
        sep = backbone.get('sep', False)
        if conv in ['gat', 'gt'] and sep:
            model_name = {
                'gat': 'GAT-sep',
                'gt': 'GT-sep',
            }[conv]
        else:
            model_name = {
                'gcn': 'GCN',
                'sage': 'SAGE',
                'gat': 'GAT',
                'gt': 'GT',
            }[conv]

        d_in = n_features
        d_out = 1 if n_classes is None else n_classes

        self.backbone = _Backbone(
            model_name=model_name,
            num_layers=backbone['n_blocks'],
            input_dim=d_in,
            hidden_dim=backbone['d_hidden'],
            output_dim=d_out,
            hidden_dim_multiplier=backbone.get('hidden_dim_multiplier', 1),
            num_heads=backbone.get('n_heads', 1),
            normalization=backbone.get('normalization', 'LayerNorm'),
            dropout=backbone['dropout'],
        )

    def forward(
        self, graph: dgl.DGLGraph, features: dict[str, None | Tensor]
    ) -> Tensor:
        x = []
        for value in features.values():
            if value is None:
                continue
            x.append(value)

        x = torch.cat(x, dim=1)
        x = self.backbone(graph, x)
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

    @torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    def apply_model() -> Tensor:
        outputs = model(graph, features)
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
                'optimizer': optimizer.state_dict(),
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
        report['training_max_memory_allocated_bytes'] = (
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
