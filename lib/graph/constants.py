# >>> by source


GRAPHLAND_DATASETS = [
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'tolokers-2',
    'city-reviews',
    'artnet-exp',
    'web-fraud',
    'hm-prices',
    'avazu-ctr',
    'city-roads-M',
    'city-roads-L',
    'twitch-views',
    'artnet-views',
    'web-traffic',
]

PYG_DATASETS = [
    'roman-empire',
    'amazon-ratings',
    'minesweeper',
    'tolokers',
    'questions',
    'cora',
    'citeseer',
    'pubmed',
    'coauthor-cs',
    'coauthor-physics',
    'amazon-computers',
    'amazon-photo',
    'lastfm-asia',
    'facebook',
    'flickr',
    'wiki-cs',
]

OGB_DATASETS = [
    'ogbn-arxiv',
    'ogbn-products',
]

OTHER_DATASETS = [
    'amazon-ratings-full',
]

DATASETS = [
    *GRAPHLAND_DATASETS,
    *PYG_DATASETS,
    *OGB_DATASETS,
    *OTHER_DATASETS,
]


# >>> by task


MULTICLASS_DATASETS = [
    'roman-empire',
    'amazon-ratings',
    'cora',
    'citeseer',
    'pubmed',
    'coauthor-cs',
    'coauthor-physics',
    'amazon-computers',
    'amazon-photo',
    'lastfm-asia',
    'facebook',
    'flickr',
    'wiki-cs',
    'ogbn-arxiv',
    'ogbn-products',
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'amazon-ratings-full',
]

BINCLASS_DATASETS = [
    'minesweeper',
    'tolokers',
    'questions',
    'tolokers-2',
    'city-reviews',
    'artnet-exp',
    'web-fraud',
]

REGRESSION_DATASETS = [
    'hm-prices',
    'avazu-ctr',
    'city-roads-M',
    'city-roads-L',
    'twitch-views',
    'artnet-views',
    'web-traffic',
]

assert (
    len(MULTICLASS_DATASETS) +
    len(BINCLASS_DATASETS) +
    len(REGRESSION_DATASETS)
) == len(DATASETS)  # fmt: skip


# >>> by features


HETEROPHILOUS_DATASETS = [
    'roman-empire',
    'amazon-ratings',
    'minesweeper',
    'tolokers',
    'questions',
]

PREDEFINED_SPLIT_DATASETS = [
    *HETEROPHILOUS_DATASETS,
    *GRAPHLAND_DATASETS,
    *OTHER_DATASETS,
]

HOMOGENEOUS_DATASETS = [
    *PYG_DATASETS,
    *OGB_DATASETS,
    *OTHER_DATASETS,
]

HETEROGENEOUS_DATASETS = [
    *GRAPHLAND_DATASETS,
]

NO_AMP_DATASETS = {
    'hm-categories',
    'pokec-regions',
    'web-topics',
    'web-fraud',
    'hm-prices',
    'avazu-ctr',
    'twitch-views',
    'web-traffic',
}


# >>> project datasets


BENCHMARK_DATASETS = [
    'tolokers-2',
    'city-reviews',
    'artnet-exp',
    'hm-categories',
    #
    'hm-prices',
    'avazu-ctr',
    'city-roads-M',
    'city-roads-L',
    'twitch-views',
    'artnet-views',
]
