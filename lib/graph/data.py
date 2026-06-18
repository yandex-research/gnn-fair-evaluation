import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, NotRequired, TypedDict, TypeVar, cast

import dgl
import numpy as np
import pandas as pd
import sklearn.impute
import sklearn.preprocessing
import torch
import yaml
from ogb.nodeproppred import DglNodePropPredDataset as OGBDataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric import datasets as pyg_datasets
from torch_geometric.data import Data as PygData

from lib.data import _SCORE_SHOULD_BE_MAXIMIZED
from lib.graph import constants
from lib.graph.util import Setting
from lib.metrics import calculate_metrics as calculate_metrics_
from lib.util import DATA_PARTS, KWArgs, PartKey, PredictionType, Score, TaskType


class GraphLandInfo(TypedDict):
    dataset_name: str
    task: str
    metric: str
    numerical_features_names: list[str]
    categorical_features_names: list[str]
    fraction_features_names: list[str]
    target_name: str
    graph_is_directed: bool
    has_unlabeled_nodes: bool
    has_nans_in_numerical_features: bool


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as file:
        data = yaml.safe_load(file)
    return data


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as file:
        data = json.load(file)
    return data


def _mask_labeled_nodes(labels: np.ndarray, masks: dict[PartKey, np.ndarray]) -> None:
    mask_labeled = ~np.isnan(labels)
    for part in DATA_PARTS:
        masks[part] = masks[part] & mask_labeled


class GraphData(TypedDict):
    name: str
    graph: dgl.DGLGraph
    labels: np.ndarray
    masks: dict[PartKey, np.ndarray]
    num_features: None | np.ndarray  # typically requires scaling
    cat_features: None | np.ndarray  # requires encoding
    frac_features: None | np.ndarray  # may not require any processing


GRAPH_FEATURE_KEYS = [f'{key}_features' for key in ['num', 'cat', 'frac']]


def _load_graphland_data(
    path: str | Path,
    *,
    internal_split_name: Literal['RL', 'RH', 'TH'],
    **kwargs,
) -> GraphData:
    del kwargs
    path = Path(path).resolve()
    name = path.name

    if name not in constants.GRAPHLAND_DATASETS:
        raise ValueError(f'Unknown dataset {name=}!')

    # >>> Load info
    info = cast(GraphLandInfo, load_yaml(path / 'info.yaml'))

    # >>> Load features
    features_df = pd.read_csv(path / 'features.csv', index_col=0).astype(np.float32)

    # >>> Drop constant features
    features_df = features_df.loc[:, features_df.apply(pd.Series.nunique) != 1]
    columns_remained = list(features_df.columns)

    # >>> Load labels
    targets_df = pd.read_csv(path / 'targets.csv', index_col=0).astype(np.float32)
    labels = targets_df.values.squeeze()

    # >>> Load & prepare data split
    masks_df = pd.read_csv(
        path / f'split_masks_{internal_split_name}.csv',
        index_col=0,
    ).astype(bool)
    masks: dict[PartKey, np.ndarray] = {
        part: masks_df[part].values for part in DATA_PARTS
    }  # type: ignore
    _mask_labeled_nodes(labels, masks)

    # >>> Separate features of different types
    # NOTE: fraction features in GraphLand are treated as numerical features
    frac_features_names = [
        feature_name
        for feature_name in info['fraction_features_names']
        if feature_name in columns_remained
    ]
    frac_features = (
        features_df.loc[:, frac_features_names].values.astype(np.float32)
        if frac_features_names
        else None
    )
    if frac_features is not None:
        assert not np.isnan(frac_features).any().item(), (
            'Fraction features can not contain nans'
        )

    # NOTE: categorical features in GraphLand do not contain nans
    # so casting to integer dtype should not throw exceptions
    cat_features_names = [
        feature_name
        for feature_name in info['categorical_features_names']
        if feature_name in columns_remained
    ]
    cat_features = (
        features_df.loc[:, cat_features_names].values.astype(np.int32)
        if cat_features_names
        else None
    )

    num_features_names = [
        feature_name
        for feature_name in info['numerical_features_names']
        if feature_name not in frac_features_names and feature_name in columns_remained
    ]
    num_features = (
        features_df.loc[:, num_features_names].values.astype(np.float32)
        if num_features_names
        else None
    )

    # >>> Construct graph
    edges_df = pd.read_csv(path / 'edgelist.csv')
    edges = torch.from_numpy(edges_df.values[:, :2]).T
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=len(labels),
        idtype=torch.int32,
    )

    graph_data = {
        'name': name,
        'graph': graph,
        'labels': labels,
        'masks': masks,
        'num_features': num_features,
        'cat_features': cat_features,
        'frac_features': frac_features,
    }
    return cast(GraphData, graph_data)


def _load_pyg_data(
    path: str | Path,
    *,
    external_split_file: None | str = None,
    internal_split_index: None | int = None,
    **kwargs,
) -> GraphData:
    del kwargs
    assert (external_split_file is not None) ^ (internal_split_index is not None), (
        'Specifying internal_split_index exludes external_split_file.'
    )

    path = Path(path).resolve()
    name = path.name
    root = str(path)

    if name in [
        'roman-empire',
        'amazon-ratings',
        'minesweeper',
        'tolokers',
        'questions',
    ]:
        dataset = pyg_datasets.HeterophilousGraphDataset(name=name, root=root)

    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = pyg_datasets.Planetoid(name=name, root=root)

    elif name in ['coauthor-cs', 'coauthor-physics']:
        dataset = pyg_datasets.Coauthor(name=name.split('-')[1], root=root)

    elif name in ['amazon-computers', 'amazon-photo']:
        dataset = pyg_datasets.Amazon(name=name.split('-')[1], root=root)

    elif name == 'lastfm-asia':
        dataset = pyg_datasets.LastFMAsia(root=root)

    elif name == 'facebook':
        dataset = pyg_datasets.FacebookPagePage(root=root)

    elif name == 'flickr':
        dataset = pyg_datasets.Flickr(root=root)

    elif name == 'wiki-cs':
        dataset = pyg_datasets.WikiCS(root=root)

    else:
        raise ValueError(f'Unknown dataset {name=}!')

    data: PygData = dataset[0]  # type: ignore

    labels: np.ndarray = data.y.squeeze().numpy().astype(np.float32)  # type: ignore
    num_features: np.ndarray = data.x.numpy().astype(np.float32)  # type: ignore
    cat_features = None
    frac_features = None

    edges: Tensor = data.edge_index  # type: ignore
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=len(labels),
        idtype=torch.int32,
    )

    if external_split_file is not None:
        masks: dict[PartKey, np.ndarray] = dict(
            np.load(path / external_split_file, allow_pickle=True)
        )

    else:
        if name in constants.PREDEFINED_SPLIT_DATASETS:

            def _retrieve_data_split(split_index):
                return {
                    part: getattr(data, f'{part}_mask')[:, split_index].numpy()
                    for part in DATA_PARTS
                }

            masks: dict[PartKey, np.ndarray] = _retrieve_data_split(
                internal_split_index
            )

        else:

            def _generate_data_split(seed: int = 17):
                masks = {part: np.zeros(len(labels), dtype=bool) for part in DATA_PARTS}
                indices_train, indices_holdout = train_test_split(
                    np.arange(len(labels)),
                    test_size=0.5,
                    random_state=seed,
                    stratify=labels,
                )
                masks['train'][indices_train] = True

                indices_val, indices_test = train_test_split(
                    indices_holdout,
                    test_size=0.5,
                    random_state=seed,
                    stratify=labels[indices_holdout],
                )
                masks['val'][indices_val] = True
                masks['test'][indices_test] = True

                return masks

            masks: dict[PartKey, np.ndarray] = _generate_data_split()

    _mask_labeled_nodes(labels, masks)

    graph_data = {
        'name': name,
        'graph': graph,
        'labels': labels,
        'masks': masks,
        'num_features': num_features,
        'cat_features': cat_features,
        'frac_features': frac_features,
    }
    return cast(GraphData, graph_data)


def _load_ogb_data(
    path: str | Path,
    *,
    external_split_file: None | str = None,
    **kwargs,
) -> GraphData:
    del kwargs
    assert external_split_file is None, (
        'No external splits are available for OGB datasets!'
    )

    path = Path(path).resolve()
    name = path.name
    root = str(path)

    if name not in constants.OGB_DATASETS:
        raise ValueError(f'Unknown dataset {name=}!')

    dataset = OGBDataset(name, root=root)
    graph: dgl.DGLGraph = dataset[0][0]
    features: Tensor = graph.ndata['feat']  # type: ignore
    labels: Tensor = dataset[0][1]  # type: ignore

    labels = labels.squeeze().numpy().astype(np.float32)  # type: ignore
    num_features = features.numpy().astype(np.float32)
    cat_features = None
    frac_features = None

    del graph.ndata['feat']
    graph = graph.int()

    data_split_: dict = dataset.get_idx_split()  # type: ignore
    masks: dict[PartKey, np.ndarray] = dict()
    for part in DATA_PARTS:
        mask = np.zeros(len(labels), dtype=bool)
        # NOTE: quick fix for validation part name
        mask[data_split_[part if part != 'val' else 'valid']] = True
        masks[part] = mask

    _mask_labeled_nodes(labels, masks)  # type: ignore

    graph_data = {
        'name': name,
        'graph': graph,
        'labels': labels,
        'masks': masks,
        'num_features': num_features,
        'cat_features': cat_features,
        'frac_features': frac_features,
    }
    return cast(GraphData, graph_data)


def _load_other_data(
    path: str | Path,
    *,
    internal_split_name: Literal['RL', 'RH', 'TH'],
    **kwargs,
) -> GraphData:
    del kwargs

    path = Path(path).resolve()
    name = path.name

    if path.name not in constants.OTHER_DATASETS:
        raise ValueError(f'Unknown dataset {name=}!')

    features: np.ndarray = np.load(path / 'features.npy', allow_pickle=True)
    num_features = features.astype(np.float32)
    cat_features = None
    frac_features = None

    labels: np.ndarray = np.load(path / 'targets.npy', allow_pickle=True)
    labels = labels.squeeze().astype(np.float32)

    edges_npy: np.ndarray = np.load(path / 'edgelist.npy', allow_pickle=True)
    edges = torch.from_numpy(edges_npy).T
    graph = dgl.graph(
        data=(edges[0], edges[1]),
        num_nodes=len(labels),
        idtype=torch.int32,
    )

    masks_npz = np.load(path / f'split_{internal_split_name}.npz', allow_pickle=True)
    masks: dict[PartKey, np.ndarray] = {
        part: masks_npz[f'{part}_mask'] for part in DATA_PARTS
    }

    _mask_labeled_nodes(labels, masks)

    graph_data = {
        'name': name,
        'graph': graph,
        'labels': labels,
        'masks': masks,
        'num_features': num_features,
        'cat_features': cat_features,
        'frac_features': frac_features,
    }
    return cast(GraphData, graph_data)


def load_data(path: str | Path, **data_params) -> GraphData:
    """
    Load data, drop constant categorical features,
    move binary categorical features to fraction features, if applicable.
    """

    path = Path(path).resolve()
    name = path.name

    if name in constants.GRAPHLAND_DATASETS:
        _load_data_fn = _load_graphland_data

    elif name in constants.PYG_DATASETS:
        _load_data_fn = _load_pyg_data

    elif name in constants.OGB_DATASETS:
        _load_data_fn = _load_ogb_data

    elif name in constants.OTHER_DATASETS:
        _load_data_fn = _load_other_data

    else:
        raise ValueError(f'Unknown {name=}')

    data = _load_data_fn(path, **data_params)

    cat_features = data['cat_features']
    if cat_features is not None:
        # >>> Drop constant features
        mask = np.array([len(np.unique(x)) > 1 for x in cat_features.T])
        assert mask.any().item()
        cat_features = cat_features[:, mask]
        data['cat_features'] = cat_features

    frac_features = data['frac_features']
    if frac_features is not None:
        # >>> Transform binary fraction features to categorical features
        mask = np.array([np.all((x == 0.0) | (x == 1.0)) for x in frac_features.T])
        if mask.any().item():
            bin_features = frac_features[:, mask].astype(np.float32)
            cat_features = data['cat_features']
            cat_features = (
                np.concatenate([cat_features, bin_features], axis=1)
                if cat_features is not None
                else bin_features
            )
            data['cat_features'] = cat_features
            data['frac_features'] = (
                None if mask.all().item() else frac_features[:, ~mask]
            )

    # >>> Post-process graph
    graph = data['graph']
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    data['graph'] = graph  # type: ignore

    return data


@dataclass(frozen=True)
class GraphTask:
    labels: np.ndarray
    masks: dict[PartKey, np.ndarray]
    type_: TaskType
    setting: Setting
    score: Score

    @classmethod
    def from_dir(
        cls, path: str | Path, setting: str | Setting, **data_params
    ) -> 'GraphTask':
        data = load_data(path, **data_params)
        task_type = _get_task_type(data['name'])
        score = _get_score(task_type)
        return GraphTask(
            labels=data['labels'],
            masks=data['masks'],
            type_=task_type,
            setting=Setting(setting),
            score=score,
        )

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    @property
    def is_transductive(self) -> bool:
        return self.setting == Setting.TRANSDUCTIVE

    def compute_n_classes(self) -> int:
        assert self.is_classification
        mask_labeled = ~np.isnan(self.labels)
        n_classes = len(np.unique(self.labels[mask_labeled]))
        return n_classes

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        # NOTE: such inconsistency between labels and predictions
        # makes `GraphTask.calculate_metrics` compatible with other scripts
        labels = {part: self.labels[self.masks[part]] for part in DATA_PARTS}
        metrics = {
            part: calculate_metrics_(
                labels[part], predictions[part], self.type_, prediction_type
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics


T = TypeVar('T', np.ndarray, Tensor)


def _get_task_type(name: str) -> TaskType:
    if name in constants.BINCLASS_DATASETS:
        return TaskType.BINCLASS

    elif name in constants.MULTICLASS_DATASETS:
        return TaskType.MULTICLASS

    elif name in constants.REGRESSION_DATASETS:
        return TaskType.REGRESSION

    else:
        raise ValueError(f'Unknown dataset {name=}!')


def _get_score(task_type: TaskType) -> Score:
    return {
        TaskType.BINCLASS: Score.AP,
        TaskType.MULTICLASS: Score.ACCURACY,
        TaskType.REGRESSION: Score.R2,
    }[task_type]


@dataclass
class GraphDataset(Generic[T]):  # noqa: UP046
    data: GraphData
    task: GraphTask

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        setting: str | Setting,
        score: None | str | Score = None,
        **params,
    ) -> 'GraphDataset[np.ndarray]':
        data = load_data(path, **params)
        task_type = _get_task_type(data['name'])
        task = GraphTask(
            labels=data['labels'],
            masks=data['masks'],
            type_=task_type,
            setting=Setting(setting),
            score=Score(score) if score is not None else _get_score(task_type),
        )
        return GraphDataset(data, task)

    def _is_numpy(self) -> bool:
        for key in GRAPH_FEATURE_KEYS:
            if self.data[key] is not None:
                return isinstance(self.data[key], np.ndarray)
        raise ValueError('No features are available!')

    @property
    def is_heterogeneous(self) -> bool:
        return self.data['name'] in constants.HETEROGENEOUS_DATASETS

    @property
    def n_num_features(self) -> int:
        # NOTE: homogeneous features are treated as numerical
        num_features = self.data['num_features']
        return num_features.shape[1] if num_features is not None else 0

    @property
    def n_cat_features(self) -> int:
        assert self.is_heterogeneous
        cat_features = self.data['cat_features']
        return cat_features.shape[1] if cat_features is not None else 0

    @property
    def n_frac_features(self) -> int:
        assert self.is_heterogeneous
        frac_features = self.data['frac_features']
        return frac_features.shape[1] if frac_features is not None else 0

    @property
    def n_features(self) -> int:
        return (
            (self.n_num_features + self.n_cat_features + self.n_frac_features)
            if self.is_heterogeneous
            else self.n_num_features
        )

    def size(self, part: None | PartKey = None) -> int:
        return (
            self.data['masks'][part].sum().item()
            if part is not None
            else len(self.data['labels'])
        )

    def to_torch(self, device: None | str | torch.device) -> 'GraphDataset[Tensor]':
        data_casted = cast(GraphData, dict())
        for key, value in self.data.items():
            data_casted[key] = (
                {
                    subkey: torch.as_tensor(subvalue).to(device)
                    for subkey, subvalue in value.items()
                }
                if isinstance(value, dict)
                else torch.as_tensor(value).to(device)
                if isinstance(value, np.ndarray)
                else value.to(device)
                if isinstance(value, dgl.DGLGraph)
                else value
            )
        return GraphDataset(data_casted, self.task)


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    QUANTILE_NORMAL = 'quantile-normal'
    QUANTILE_UNIFORM = 'quantile-uniform'


def transform_num_features(
    features: np.ndarray,
    masks: dict[PartKey, np.ndarray],
    *,
    setting: str | Setting,
    policy: None | str | NumPolicy,
    impute_first: bool = False,
    seed: None | int,
) -> np.ndarray:
    if policy is None:
        return features

    policy = NumPolicy(policy)
    # NOTE: throws exception in case of unknown value

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks['train'], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks['train']
        # NOTE: this is not correct for inductive setting, as train graph
        # can have unlabeled nodes that must be taken into account
    )

    def _impute(features: np.ndarray, mask_seen: np.ndarray) -> np.ndarray:
        imputer = sklearn.impute.SimpleImputer()
        imputer.fit(features[mask_seen])
        features = imputer.transform(features)  # type: ignore
        return features

    if impute_first:
        features = _impute(features, mask_seen)

    # NOTE: reserve a separate variable to avoid modifications in the original features
    features_seen = features[mask_seen]

    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()

    else:
        distribution = 'normal' if policy == NumPolicy.QUANTILE_NORMAL else 'uniform'
        assert seed is not None
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(features_seen.shape[0] // 30, 1000), 10),
            output_distribution=distribution,
            subsample=1_000_000_000,
            random_state=seed,
        )
        features_seen = features_seen + np.random.RandomState(seed).normal(
            0.0, 1e-5, features_seen.shape
        ).astype(features_seen.dtype)

    normalizer.fit(features_seen)
    features_trasformed = normalizer.transform(features)

    if not impute_first:
        features_trasformed = _impute(features_trasformed, mask_seen)

    return features_trasformed  # type: ignore


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


def transform_cat_features(
    features: np.ndarray,
    masks: dict[PartKey, np.ndarray],
    *,
    setting: str | Setting,
    policy: None | str | CatPolicy,
) -> np.ndarray:
    if policy is None:
        return features

    # NOTE: throws exception in case of unknown value
    policy = CatPolicy(policy)

    setting = Setting(setting)
    mask_seen = (
        np.ones_like(masks['train'], dtype=bool)
        if setting == Setting.TRANSDUCTIVE
        else masks['train']
        # NOTE: this is not correct for inductive setting, as train graph
        # can have unlabeled nodes that must be taken into account
    )

    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.int32,
    ).fit(features[mask_seen])
    features_encoded = encoder.transform(features)

    if policy == CatPolicy.ORDINAL:
        return features_encoded

    if policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            drop='if_binary',
            sparse_output=False,
            handle_unknown='ignore',
            dtype=np.float32,
        )

    encoder.fit(features_encoded[mask_seen])
    features_transformed = encoder.transform(features_encoded)
    return features_transformed  # type: ignore


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    labels: np.ndarray,
    masks: dict[PartKey, np.ndarray],
) -> tuple[np.ndarray, RegressionLabelStats]:
    assert labels.dtype == np.float32

    labels_seen = labels[masks['train']]
    mean = float(labels_seen.mean())
    std = float(labels_seen.std())

    labels_standardized = (labels - mean) / std
    regression_stats = RegressionLabelStats(mean=mean, std=std)
    return labels_standardized, regression_stats


def _nfa_reduce_single_hop(
    g: dgl.DGLGraph,
    x: np.ndarray,
    *,
    mode: str,
    hop: int = 1,
) -> np.ndarray:
    assert mode in ['mean', 'max', 'min']
    assert hop > 0
    x_ = torch.tensor(x)

    not_nan_mask = ~torch.isnan(x_)
    not_nan_size = not_nan_mask.float()
    for _ in range(hop):
        not_nan_size = dgl.ops.copy_u_sum(g, not_nan_size)  # type: ignore

    neutral_value = {
        'mean': 0.0,
        'max': -torch.inf,
        'min': +torch.inf,
    }[mode]

    operation = {
        'mean': dgl.ops.copy_u_sum,  # type: ignore
        'max': dgl.ops.copy_u_max,  # type: ignore
        'min': dgl.ops.copy_u_min,  # type: ignore
    }[mode]

    denominator = {
        'mean': not_nan_size,
        'max': (not_nan_size > 0.0).float(),
        'min': (not_nan_size > 0.0).float(),
    }[mode]

    numerator = torch.where(not_nan_mask, x_, neutral_value)
    for _ in range(hop):
        numerator = operation(g, numerator)

    x_reduced = torch.where(denominator != 0.0, numerator / denominator, torch.nan)
    return x_reduced.numpy()


def _nfa_reduce(
    g: dgl.DGLGraph,
    x: np.ndarray,
    *,
    mode: str,
    weights: list[float],
) -> np.ndarray:
    values = [x]
    for hop in range(1, len(weights)):
        values.append(_nfa_reduce_single_hop(g, x, mode=mode, hop=hop))
    return (np.array(weights) * np.stack(values, axis=-1)).sum(-1).astype(np.float32)


def apply_nfa(
    dataset: GraphDataset[np.ndarray],
    num_modes: list[str] = ['mean', 'max', 'min'],
    weights: list[float] = [0.0, 1.0],
) -> dict[str, None | np.ndarray]:
    nfa: dict[str, list[np.ndarray]] = {key: [] for key in GRAPH_FEATURE_KEYS}
    data = dataset.data
    graph = data['graph']

    if data['num_features'] is not None:
        for mode in num_modes:
            nfa['num_features'].append(
                _nfa_reduce(graph, data['num_features'], mode=mode, weights=weights)
            )

    if data['cat_features'] is not None:
        cat_features_transformed = transform_cat_features(
            data['cat_features'],
            data['masks'],
            setting=dataset.task.setting,
            policy=CatPolicy('one-hot'),
        )
        nfa['frac_features'].append(
            _nfa_reduce(graph, cat_features_transformed, mode='mean', weights=weights)
        )

    if data['frac_features'] is not None:
        nfa['frac_features'].append(
            _nfa_reduce(graph, data['frac_features'], mode='mean', weights=weights)
        )

    # NOTE: categorical features never occur here, so the corresponding value is None
    return {
        key: np.concatenate(nfa[key], axis=1) if nfa[key] else None
        for key in GRAPH_FEATURE_KEYS
    }


def merge_features(data: GraphData, *args: dict[str, None | np.ndarray]) -> GraphData:
    container: dict[str, list[np.ndarray]] = {key: [] for key in GRAPH_FEATURE_KEYS}
    for item in args:
        assert set(item.keys()) <= set(container.keys())
        # NOTE: some feature keys may not occur
        for key in GRAPH_FEATURE_KEYS:
            if item.get(key, None) is not None:
                container[key].append(item[key])  # type: ignore

    for key in GRAPH_FEATURE_KEYS:
        features = []
        if data[key] is not None:
            features.append(data[key])
        features.extend(container[key])
        data[key] = np.concatenate(features, axis=1) if features else None
    return data


@torch.no_grad()
def compute_degrees(graph: dgl.DGLGraph, log: bool = True) -> Tensor:
    degrees = 0.5 * (graph.in_degrees() + graph.out_degrees())
    if log:
        degrees = torch.log(1 + degrees)
    return degrees.unsqueeze(-1)


@torch.no_grad()
def compute_pagerank(
    graph: dgl.DGLGraph,
    alpha: float = 0.85,
    max_iterations: int = 100,
    tol: float = 1e-6,
    reset_nodes: None | Tensor = None,
    log: bool = True,
) -> Tensor:
    assert alpha >= 0.5, 'Make sure that you provde alpha, not 1 - alpha!'

    n_nodes = graph.num_nodes()
    dist = torch.ones(n_nodes, device=graph.device) / n_nodes
    degrees = graph.out_degrees().float()

    if reset_nodes is None:
        reset_dist = (1 - alpha) / n_nodes
    else:
        reset_dist = torch.zeros(n_nodes, dtype=torch.float32)
        reset_dist[reset_nodes] = 1.0
        reset_dist /= reset_dist.sum()
        reset_dist *= 1 - alpha

    for _ in range(max_iterations):
        prev_dist = dist.clone()
        dist = dgl.ops.copy_u_sum(graph, dist / degrees)  # type: ignore
        dist = alpha * dist + reset_dist
        err = torch.abs(dist - prev_dist).sum().item()
        if err < tol:
            break

    if log:
        dist = torch.log(tol + dist)
    return dist.unsqueeze(-1)


def get_structural_encodings(graph: dgl.DGLGraph, *, name: str, k: int) -> np.ndarray:
    if name == 'lappe':
        encodings: np.ndarray = dgl.lap_pe(graph, k).numpy()  # type: ignore
        return encodings

    if name == 'randn':
        return np.random.randn(graph.num_nodes(), k).astype(np.float32)

    raise ValueError(f'Unknown type of structural encodings: {name}.')


class TransformConfig(TypedDict):
    seed: int
    labels: bool
    features: NotRequired[KWArgs]
    nfa: NotRequired[KWArgs]


def transform_features(
    data: dict[str, None | np.ndarray],
    *,
    task: GraphTask,
    num_policy: None | str | NumPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    frac_policy: None | str | NumPolicy = None,
    impute_first: bool = False,
    seed: int = 0,
) -> dict[str, None | np.ndarray]:
    for key, policy in zip(
        GRAPH_FEATURE_KEYS,
        [num_policy, cat_policy, frac_policy],
    ):
        if data[key] is None:
            assert policy is None
            continue

        if key == 'cat_features':
            data[key] = transform_cat_features(  # type: ignore
                data[key],  # type: ignore
                task.masks,
                setting=task.setting,
                policy=policy,  # type: ignore
            )
        else:
            data[key] = transform_num_features(
                data[key],  # type: ignore
                task.masks,
                setting=task.setting,
                policy=policy,  # type: ignore
                impute_first=impute_first,
                seed=seed,
            )

    return data
