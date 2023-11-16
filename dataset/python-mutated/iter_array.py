from __future__ import annotations
import itertools
import random
import numpy as np
from river import base

def iter_array(X: np.ndarray, y: np.ndarray | None=None, feature_names: list[base.typing.FeatureName] | None=None, target_names: list[base.typing.FeatureName] | None=None, shuffle: bool=False, seed: int | None=None) -> base.typing.Stream:
    if False:
        for i in range(10):
            print('nop')
    'Iterates over the rows from an array of features and an array of targets.\n\n    This method is intended to work with `numpy` arrays, but should also work with Python lists.\n\n    Parameters\n    ----------\n    X\n        A 2D array of features. This can also be a 1D array of strings, which can be the case if\n        you\'re working with text.\n    y\n        An optional array of targets.\n    feature_names\n        An optional list of feature names. The features will be labeled with integers if no names\n        are provided.\n    target_names\n        An optional list of output names. The outputs will be labeled with integers if no names are\n        provided. Only applies if there are multiple outputs, i.e. if `y` is a 2D array.\n    shuffle\n        Indicates whether or not to shuffle the input arrays before iterating over them.\n    seed\n        Random seed used for shuffling the data.\n\n    Examples\n    --------\n\n    >>> from river import stream\n    >>> import numpy as np\n\n    >>> X = np.array([[1, 2, 3], [11, 12, 13]])\n    >>> Y = np.array([True, False])\n\n    >>> dataset = stream.iter_array(\n    ...     X, Y,\n    ...     feature_names=[\'x1\', \'x2\', \'x3\']\n    ... )\n    >>> for x, y in dataset:\n    ...     print(x, y)\n    {\'x1\': 1, \'x2\': 2, \'x3\': 3} True\n    {\'x1\': 11, \'x2\': 12, \'x3\': 13} False\n\n    This also works with a array of texts:\n\n    >>> X = ["foo", "bar"]\n    >>> dataset = stream.iter_array(\n    ...     X, Y,\n    ...     feature_names=[\'x1\', \'x2\', \'x3\']\n    ... )\n    >>> for x, y in dataset:\n    ...     print(x, y)\n    foo True\n    bar False\n\n    '
    if isinstance(X[0], str):

        def handle_features(x):
            if False:
                while True:
                    i = 10
            return x
    else:
        feature_names = list(range(len(X[0]))) if feature_names is None else feature_names

        def handle_features(x):
            if False:
                return 10
            return dict(zip(feature_names, xi))
    multioutput = y is not None and (not np.isscalar(y[0]))
    if multioutput and target_names is None:
        target_names = list(range(len(y[0])))
    rng = random.Random(seed)
    if shuffle:
        order = rng.sample(range(len(X)), k=len(X))
        X = X[order]
        y = y if y is None else y[order]
    if multioutput:
        for (xi, yi) in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
            yield (handle_features(xi), dict(zip(target_names, yi)))
    else:
        for (xi, yi) in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
            yield (handle_features(xi), yi)