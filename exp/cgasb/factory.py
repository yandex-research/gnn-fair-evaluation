import pathlib
from itertools import product

import lib.graph.constants

SELECTED_DATASETS = list(lib.graph.constants.BENCHMARK_DATASETS)
SELECTED_CONVS = ['gcn', 'sage', 'gat', 'gt']

NUM_POLICY_OPTIONS = {
    'hm-categories': 'quantile-normal',
    'tolokers-2': 'quantile-normal',
    'city-reviews': 'quantile-normal',
    'artnet-exp': 'quantile-normal',
    'hm-prices': 'quantile-normal',
    'city-roads-M': 'quantile-normal',
    'artnet-views': None,  # NA
    'avazu-ctr': None,  # NA
    'city-roads-L': 'quantile-normal',
    'twitch-views': None,  # NA
}


CAT_POLICY_OPTIONS = {
    'hm-categories': 'one-hot',
    'tolokers-2': 'one-hot',
    'city-reviews': 'one-hot',
    'artnet-exp': 'one-hot',
    'hm-prices': 'one-hot',
    'city-roads-M': 'one-hot',
    'artnet-views': 'one-hot',
    'avazu-ctr': None,  # NA
    'city-roads-L': 'one-hot',
    'twitch-views': 'one-hot',
}

FRAC_POLICY_OPTIONS = {
    'hm-categories': None,
    'tolokers-2': None,
    'city-reviews': None,
    'artnet-exp': None,
    'hm-prices': None,
    'city-roads-M': None,
    'artnet-views': 'quantile-normal',
    'avazu-ctr': 'quantile-normal',
    'city-roads-L': None,
    'twitch-views': None,
}


def main() -> None:
    root = pathlib.Path(__file__).parent.resolve()
    function_ = 'bin.cgasb.main'

    for conv, dataset in product(SELECTED_CONVS, SELECTED_DATASETS):
        # >>> Data
        _data_config = {
            'path': f'data/{dataset}',
            'setting': 'transductive',
        }
        _transform_config = {
            'seed': 0,
            'labels': False,
        }
        _features_config = {}

        if dataset in lib.graph.constants.HETEROGENEOUS_DATASETS:
            _data_config['internal_split_name'] = 'RL'
            for key, policy in zip(
                ['num', 'cat', 'frac'],
                [
                    NUM_POLICY_OPTIONS[dataset],
                    CAT_POLICY_OPTIONS[dataset],
                    FRAC_POLICY_OPTIONS[dataset],
                ],
            ):
                if policy is not None:
                    _features_config[f'{key}_policy'] = policy

            if dataset in lib.graph.constants.REGRESSION_DATASETS:
                _transform_config['labels'] = True

        elif dataset in lib.graph.constants.OTHER_DATASETS:
            _data_config['internal_split_name'] = 'RL'

        else:
            if dataset in lib.graph.constants.PYG_DATASETS:
                _data_config['external_split_file'] = 'split.npz'

        if _features_config:
            _features_config['impute_first'] = True
            _transform_config['features'] = _features_config

        # >>> Optimizer
        _optimizer_config = {
            'type': 'AdamW',
            'lr': [
                '_tune_',
                'loguniform',
                3e-5,
                1e-2,
            ],
            'weight_decay': [
                '_tune_',
                'loguniform',
                1e-4,
                1e-0,
            ],
        }

        # >>> Backbone
        _backbone_config = {
            'conv': conv,
            'n_blocks': [
                '_tune_',
                'int',
                1,
                10,
                1,
            ],
            'd_hidden': [
                '_tune_',
                'int',
                96,
                768,
                32,
            ],
            'dropout': [
                '_tune_',
                'discrete_uniform',
                0.0,
                0.5,
                0.05,
            ],
            'pre_linear': [
                '_tune_',
                'categorical',
                [False, True],
            ],
            'res': [
                '_tune_',
                'categorical',
                [False, True],
            ],
            'norm': [
                '_tune_',
                'categorical',
                ['none', 'ln', 'bn'],
            ],
        }

        if conv in ['gat', 'gt']:
            _backbone_config |= {
                'log2_n_heads': [
                    '_tune_',
                    'int',
                    1,
                    3,
                    1,
                ],
            }

        # >>> Model
        _model_config = {
            'backbone': _backbone_config,
        }

        # >>> Space
        _space_config = {
            'seed': 0,
            'n_steps': 3000,
            'patience': 1000,
            'optimizer': _optimizer_config,
            'data': _data_config,
            'transform': _transform_config,
            'model': _model_config,
            'save_checkpoint': False,
        }

        if dataset not in lib.graph.constants.NO_AMP_DATASETS:
            _space_config['amp_dtype'] = 'bfloat16'

        config = {
            'seed': 0,
            'function': function_,
            'n_trials': 100,
            'sampler': {
                'n_startup_trials': 25,
            },
            'space': _space_config,
        }

        path = root / conv / dataset
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / 'tuning'
        lib.dump_config(config_path, config, force=True)


if __name__ == '__main__':
    main()
