from functools import partial

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import ops
from torch import Tensor

# >>> Normalizations


class LayerNorm(nn.Module):
    def __init__(self, d_hidden: int, affine: bool = True):
        super().__init__()
        self.module = nn.LayerNorm(d_hidden, elementwise_affine=affine, bias=affine)

    def forward(self, x):
        return self.module(x)


class BatchNorm(nn.Module):
    def __init__(self, d_hidden: int, affine: bool = True):
        super().__init__()
        self.module = nn.BatchNorm1d(d_hidden, affine=affine, track_running_stats=False)

    def forward(self, x):
        return self.module(x)


class RMSNorm(nn.Module):
    def __init__(self, d_hidden: int, affine: bool = True):
        super().__init__()
        self.module = nn.RMSNorm(d_hidden, elementwise_affine=affine)

    def forward(self, x):
        return self.module(x)


class DynamicTanhNorm(nn.Module):
    def __init__(self, d_hidden: int, alpha_init: float = 0.5, affine: bool = True):
        super().__init__()
        self.affine = affine
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(d_hidden))
            self.beta = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.alpha * x)
        if self.affine:
            x = self.gamma * x + self.beta
        return x


NORMALIZATIONS = {
    'none': nn.Identity,
    'layer': LayerNorm,
    'batch': BatchNorm,
    'rms': RMSNorm,
    'dyntanh': DynamicTanhNorm,
    'layer-non-affine': partial(LayerNorm, affine=False),
    'batch-non-affine': partial(BatchNorm, affine=False),
    'rms-non-affine': partial(RMSNorm, affine=False),
    'dyntanh-non-affine': partial(DynamicTanhNorm, affine=False),
}


# >>> Activations


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        d = x.shape[1] // 2
        x1, x2 = x[:, :d], x[:, d:]
        x = x1 * F.gelu(x2)
        return x


ACTIVATIONS = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'leaky': partial(nn.LeakyReLU, negative_slope=0.2),
    'gelu': nn.GELU,
    'geglu': GEGLU,
}


# >>> Aggregation modules


class IdentityAggregation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        return x


class MeanAggregation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        return ops.copy_u_mean(graph, x)  # type: ignore


class MaxAggregation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        x = ops.copy_u_max(graph, x)  # type: ignore
        x[x.isinf()] = 0.0

        return x


class GCNAggregation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        in_degrees = graph.in_degrees().float()
        out_degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, out_degrees, in_degrees)  # type: ignore
        degree_edge_products[degree_edge_products == 0.0] = 1.0
        norm_coefs = 1.0 / degree_edge_products.sqrt()

        x = ops.u_mul_e_sum(graph, x, norm_coefs)  # type: ignore

        return x


def _check_heads_params_consistency(d: int, n_heads: int) -> None:
    if d % n_heads != 0:
        raise ValueError('Dimension mismatch: d should be a multiple of n_heads.')


def edge_softmax(graph, x):
    """
    A simple implementation of edge softmax.
    It is here as an example, one can use `dgl.ops.edge_softmax` instead.
    x should have shape [num_edges, num_heads].
    Note that `dgl.ops.edge_softmax` can also accept inputs of shape
    [num_edges, num_heads, 1], which is NOT supported by this function.
    """
    x_max = ops.copy_e_max(graph, x)  # type: ignore
    x_max[x_max.isinf()] = 0
    x = ops.e_sub_v(graph, x, x_max)  # type: ignore

    x = torch.exp(x)
    x_sum = ops.copy_e_sum(graph, x)  # type: ignore
    x = ops.e_div_v(graph, x, x_sum)  # type: ignore

    return x


def edge_softmax_with_sink(graph, x, sink=0):
    """
    A modification of edge softmax that adds :math:`exp(sink)` to the denominator,
    thus allowing attention to not attend to anything.
    x should have shape [num_edges, num_heads].
    sink can either have shape [1,] (or be a scalar), which corresponds to adding
    the same value for each head and each node, or have shape [num_heads,],
    which corresponds to adding a different value for each head
    (but the same for each node), or have shape [num_nodes, num_heads],
    which corresponds to adding a different value for each node and each head.
    By default, sink is 0, which corresponds to adding 1 to the denominator of softmax,
    as proposed in https://www.evanmiller.org/attention-is-off-by-one.html.
    """
    x_max = ops.copy_e_max(graph, x)  # type: ignore
    x_max[x_max.isinf()] = 0
    x = ops.e_sub_v(graph, x, x_max)  # type: ignore
    sink = sink - x_max

    x = torch.exp(x)
    sink = torch.exp(sink)
    x_sum = ops.copy_e_sum(graph, x)  # type: ignore
    x_sum += sink
    x = ops.e_div_v(graph, x, x_sum)  # type: ignore

    return x


class AttnGATAggregation(nn.Module):
    def __init__(self, d_hidden: int, n_heads: int, **kwargs):
        super().__init__()
        _check_heads_params_consistency(d_hidden, n_heads)
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.attn_linear_u = nn.Linear(d_hidden, n_heads)
        self.attn_linear_v = nn.Linear(d_hidden, n_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)  # type: ignore
        attn_scores = self.attn_act(attn_scores)
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.d_head, self.n_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)  # type: ignore
        x = x.reshape(-1, self.d_hidden)

        return x


class AttnGATSinkAggregation(nn.Module):
    def __init__(self, d_hidden: int, n_heads: int, **kwargs):
        super().__init__()
        _check_heads_params_consistency(d_hidden, n_heads)
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.attn_linear_u = nn.Linear(d_hidden, n_heads)
        self.attn_linear_v = nn.Linear(d_hidden, n_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)  # type: ignore
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax_with_sink(graph, attn_scores)

        x = x.reshape(-1, self.d_head, self.n_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)  # type: ignore
        x = x.reshape(-1, self.d_hidden)

        return x


class AttnTrfAggregation(nn.Module):
    def __init__(self, d_hidden: int, n_heads: int, **kwargs):
        super().__init__()
        _check_heads_params_consistency(d_hidden, n_heads)
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.attn_scores_multiplier = 1.0 / torch.tensor(self.d_head).sqrt()
        self.attn_qkv_linear = nn.Linear(d_hidden, d_hidden * 3)

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.unflatten(dim=1, sizes=(3, self.n_heads, self.d_head))
        queries, keys, values = qkvs[:, 0], qkvs[:, 1], qkvs[:, 2]

        attn_scores = ops.u_dot_v(graph, keys, queries) * self.attn_scores_multiplier  # type: ignore
        attn_probs = ops.edge_softmax(graph, attn_scores)

        x = ops.u_mul_e_sum(graph, values, attn_probs)  # type: ignore
        x = x.flatten(start_dim=1, end_dim=2)

        return x


class AttnTrfSinkAggregation(nn.Module):
    def __init__(self, d_hidden: int, n_heads: int, **kwargs):
        super().__init__()
        _check_heads_params_consistency(d_hidden, n_heads)
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.attn_scores_multiplier = 1.0 / torch.tensor(self.d_head).sqrt()
        self.attn_qkv_linear = nn.Linear(d_hidden, d_hidden * 3)

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        qkvs = self.attn_qkv_linear(x)
        qkvs = qkvs.unflatten(dim=1, sizes=(3, self.n_heads, self.d_head))
        queries, keys, values = qkvs[:, 0], qkvs[:, 1], qkvs[:, 2]

        attn_scores = ops.u_dot_v(graph, keys, queries) * self.attn_scores_multiplier  # type: ignore
        attn_probs = edge_softmax_with_sink(  #
            graph, attn_scores.squeeze(-1)
        ).unsqueeze(-1)

        x = ops.u_mul_e_sum(graph, values, attn_probs)  # type: ignore
        x = x.flatten(start_dim=1, end_dim=2)

        return x


AGGREGATIONS = {
    'identity': IdentityAggregation,
    'mean': MeanAggregation,
    'max': MaxAggregation,
    'gcn': GCNAggregation,
    'attn-gat': AttnGATAggregation,
    'attn-gat-sink': AttnGATSinkAggregation,
    'attn-trf': AttnTrfAggregation,
    'attn-trf-sink': AttnTrfSinkAggregation,
}


class MultiAggregation(nn.Module):
    def __init__(self, aggregations: list[str], **kwargs):
        super().__init__()
        self.aggregations = nn.ModuleList(
            AGGREGATIONS[aggregation](**kwargs) for aggregation in aggregations
        )
        self.n_aggregations = len(self.aggregations)

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        return torch.cat(
            [aggregation(graph, x) for aggregation in self.aggregations], axis=1
        )  # type: ignore


# >>> Backbones


class MLPModule(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        n_layers: int,
        *,
        act: str,
        norm: str,
        dropout: float,
        close: bool = False,
    ):
        if close:
            assert act != 'none'

        super().__init__()
        layers = []
        for idx in range(n_layers):
            d_in_ = d_in if idx == 0 else d_hidden
            d_out_ = d_out if idx == n_layers - 1 else d_hidden

            if (idx != n_layers - 1 or close) and act == 'geglu':
                d_out_ *= 2

            layers.append(nn.Linear(d_in_, d_out_))
            layers.append(nn.Dropout(dropout))

            if idx != n_layers - 1 or close:
                layers.append(ACTIVATIONS[act]())

            if idx != n_layers - 1:
                layers.append(NORMALIZATIONS[norm]())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class BaseBackboneBlock(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_layers: int,
        aggregations: list[str],
        *,
        act: str,
        dropout: float,
        mlp_norm: str,
        block_norm: str,
        close: bool = False,
        is_pre_norm: bool = False,
        is_post_norm: bool = False,
        is_inter_norm: bool = False,
        skip: bool = False,
        **kwargs,
    ):
        if is_pre_norm or is_post_norm:
            assert block_norm != 'none'

        super().__init__()
        self.aggregation = MultiAggregation(aggregations, d_hidden=d_hidden, **kwargs)
        self.mlp = MLPModule(
            d_in=d_hidden * self.aggregation.n_aggregations,
            d_hidden=d_hidden,
            d_out=d_hidden,
            n_layers=n_layers,
            act=act,
            norm=mlp_norm,
            dropout=dropout,
            close=close,
        )
        self.pre_norm = NORMALIZATIONS[block_norm if is_pre_norm else 'none'](d_hidden)  # fmt: skip  # noqa: E501
        self.post_norm = NORMALIZATIONS[block_norm if is_post_norm else 'none'](d_hidden)  # fmt: skip  # noqa: E501
        self.inter_norm = NORMALIZATIONS[block_norm if is_inter_norm else 'none'](d_hidden)  # fmt: skip  # noqa: E501
        self.skip = skip

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        x_res = self.pre_norm(x)
        x_res = self.aggregation(graph, x_res)
        x_res = self.mlp(x_res)
        x_res = self.inter_norm(x_res)
        x = x + x_res if self.skip else x_res
        x = self.post_norm(x)
        return x


# ... Type A


class _TypeA(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_layers: int,
        aggregations: list[str],
        *,
        act: str,
        dropout: float,
        mlp_norm: str,
        block_norm: str,
        is_pre_norm: bool = False,
        is_post_norm: bool = False,
        is_inter_norm: bool = False,
        **kwargs,
    ):
        assert n_layers == 1, f'{type(self).__name__} requires one linear layer.'
        super().__init__()
        self.block = BaseBackboneBlock(
            d_hidden=d_hidden,
            n_layers=n_layers,
            aggregations=aggregations,
            act=act,
            dropout=dropout,
            mlp_norm=mlp_norm,
            block_norm=block_norm,
            close=True,
            is_pre_norm=is_pre_norm,
            is_post_norm=is_post_norm,
            is_inter_norm=is_inter_norm,
            skip=True,
            **kwargs,
        )

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        return self.block(graph, x)


# ... Type B


class _TypeB(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_layers: int,
        aggregations: list[str],
        *,
        act: str,
        dropout: float,
        mlp_norm: str,
        block_norm: str,
        is_pre_norm: bool = False,
        is_post_norm: bool = False,
        is_inter_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.block = BaseBackboneBlock(
            d_hidden=d_hidden,
            n_layers=n_layers,
            aggregations=aggregations,
            act=act,
            dropout=dropout,
            mlp_norm=mlp_norm,
            block_norm=block_norm,
            close=False,
            is_pre_norm=is_pre_norm,
            is_post_norm=is_post_norm,
            is_inter_norm=is_inter_norm,
            skip=True,
            **kwargs,
        )

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        return self.block(graph, x)


# ... Type C


class _TypeC(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_layers: int,
        aggregations: list[str],
        *,
        act: str,
        dropout: float,
        mlp_norm: str,
        block_norm: str,
        is_pre_norm: bool = False,
        is_post_norm: bool = False,
        is_inter_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        _norm = dict(
            is_pre_norm=is_pre_norm,
            is_post_norm=is_post_norm,
            is_inter_norm=is_inter_norm,
        )
        self.block = BaseBackboneBlock(
            d_hidden=d_hidden,
            n_layers=1,
            aggregations=aggregations,
            act='none',
            dropout=dropout,
            mlp_norm=mlp_norm,
            block_norm=block_norm,
            close=False,
            skip=True,
            **_norm,
            **kwargs,
        )
        self.ffn = BaseBackboneBlock(
            d_hidden=d_hidden,
            n_layers=n_layers,
            aggregations=['identity'],
            act=act,
            dropout=dropout,
            mlp_norm=mlp_norm,
            block_norm=block_norm,
            close=False,
            skip=True,
            **_norm,
            **kwargs,
        )

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        x = self.block(graph, x)
        x = self.ffn(graph, x)
        return x


BACKBONE_BLOCKS = {
    'A': _TypeA,
    'A-pre': partial(_TypeA, is_pre_norm=True),
    'A-post': partial(_TypeA, is_post_norm=True),
    'A-sandwich': partial(_TypeA, is_pre_norm=True, is_inter_norm=True),
    #
    'B': _TypeB,
    'B-pre': partial(_TypeB, is_pre_norm=True),
    'B-post': partial(_TypeB, is_post_norm=True),
    'B-sandwich': partial(_TypeB, is_pre_norm=True, is_inter_norm=True),
    #
    'C-post': partial(_TypeC, is_post_norm=True),
    'C-sandwich': partial(_TypeC, is_pre_norm=True, is_inter_norm=True),
}


class GraphBackbone(nn.Module):
    def __init__(self, type: str, n_blocks: int, **params) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(BACKBONE_BLOCKS[type](**params))

    def forward(self, graph: dgl.DGLGraph, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(graph, x)
        return x
