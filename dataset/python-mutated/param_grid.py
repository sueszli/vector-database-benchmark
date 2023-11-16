from __future__ import annotations
import itertools
import typing
import numpy as np
from river import base
__all__ = ['expand_param_grid']

def expand_param_grid(model: base.Estimator, grid: dict) -> list[base.Estimator]:
    if False:
        i = 10
        return i + 15
    "Expands a grid of parameters.\n\n    This method can be used to generate a list of model parametrizations from a dictionary where\n    each parameter is associated with a list of possible parameters. In other words, it expands a\n    grid of parameters.\n\n    Typically, this method can be used to create copies of a given model with different parameter\n    choices. The models can then be used as part of a model selection process, such as a\n    `selection.SuccessiveHalvingClassifier` or a `selection.EWARegressor`.\n\n    The syntax for the parameter grid is quite flexible. It allows nesting parameters and can\n    therefore be used to generate parameters for a pipeline.\n\n    Parameters\n    ----------\n    model\n    grid\n        The grid of parameters to expand. The provided dictionary can be nested. The only\n        requirement is that the values at the leaves need to be lists.\n\n    Examples\n    --------\n\n    As an initial example, we can expand a grid of parameters for a single model.\n\n    >>> from river import linear_model\n    >>> from river import optim\n    >>> from river import utils\n\n    >>> model = linear_model.LinearRegression()\n\n    >>> grid = {'optimizer': [optim.SGD(.1), optim.SGD(.01), optim.SGD(.001)]}\n    >>> models = utils.expand_param_grid(model, grid)\n    >>> len(models)\n    3\n\n    >>> models[0]\n    LinearRegression (\n      optimizer=SGD (\n        lr=Constant (\n          learning_rate=0.1\n        )\n      )\n      loss=Squared ()\n      l2=0.\n      l1=0.\n      intercept_init=0.\n      intercept_lr=Constant (\n        learning_rate=0.01\n      )\n      clip_gradient=1e+12\n      initializer=Zeros ()\n    )\n\n    You can expand parameters for multiple choices like so:\n\n    >>> grid = {\n    ...     'optimizer': [\n    ...         (optim.SGD, {'lr': [.1, .01, .001]}),\n    ...         (optim.Adam, {'lr': [.1, .01, .01]})\n    ...     ]\n    ... }\n    >>> models = utils.expand_param_grid(model, grid)\n    >>> len(models)\n    6\n\n    You may specify a grid of parameters for a pipeline via nesting:\n\n    >>> from river import feature_extraction\n\n    >>> model = (\n    ...     feature_extraction.BagOfWords() |\n    ...     linear_model.LinearRegression()\n    ... )\n\n    >>> grid = {\n    ...     'BagOfWords': {\n    ...         'strip_accents': [False, True]\n    ...     },\n    ...     'LinearRegression': {\n    ...         'optimizer': [\n    ...             (optim.SGD, {'lr': [.1, .01]}),\n    ...             (optim.Adam, {'lr': [.1, .01]})\n    ...         ]\n    ...     }\n    ... }\n\n    >>> models = utils.expand_param_grid(model, grid)\n    >>> len(models)\n    8\n\n    "
    return [model.clone(params) for params in _expand_param_grid(grid)]

def _expand_param_grid(grid: dict) -> typing.Iterator[dict]:
    if False:
        return 10

    def expand_tuple(t):
        if False:
            for i in range(10):
                print('nop')
        (klass, params) = t
        if not isinstance(klass, type):
            raise ValueError(f'Expected first element to be a class, got {klass}')
        if not isinstance(params, dict):
            raise ValueError(f'Expected second element to be a dict, got {params}')
        return (klass(**combo) for combo in _expand_param_grid(params))

    def expand(k, v):
        if False:
            i = 10
            return i + 15
        if isinstance(v, tuple):
            return ((k, el) for el in expand_tuple(v))
        if isinstance(v, list) or isinstance(v, set) or isinstance(v, np.ndarray):
            combos = []
            for el in v:
                if isinstance(el, tuple):
                    for combo in expand_tuple(el):
                        combos.append((k, combo))
                else:
                    combos.append((k, el))
            return combos
        if isinstance(v, dict):
            return ((k, el) for el in _expand_param_grid(v))
        raise ValueError(f'unsupported type: {type(v)}')
    for key in grid:
        if not isinstance(key, str):
            raise ValueError(f'Expected a key of type str; got {key}')
    return (dict(el) if isinstance(el[0], tuple) else el[0] for el in itertools.product(*(expand(k, v) for (k, v) in grid.items())))