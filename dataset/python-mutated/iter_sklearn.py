from __future__ import annotations
import pandas as pd
import sklearn.utils
from river import base, stream

def iter_sklearn_dataset(dataset: sklearn.utils.Bunch, **kwargs) -> base.typing.Stream:
    if False:
        while True:
            i = 10
    "Iterates rows from one of the datasets provided by scikit-learn.\n\n    This allows you to use any dataset from [scikit-learn's `datasets` module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). For instance, you can use the `fetch_openml` function to get access to all of the\n    datasets from the OpenML website.\n\n    Parameters\n    ----------\n    dataset\n        A scikit-learn dataset.\n    kwargs\n        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.\n\n    Examples\n    --------\n\n    >>> import pprint\n    >>> from sklearn import datasets\n    >>> from river import stream\n\n    >>> dataset = datasets.load_diabetes()\n\n    >>> for xi, yi in stream.iter_sklearn_dataset(dataset):\n    ...     pprint.pprint(xi)\n    ...     print(yi)\n    ...     break\n    {'age': 0.038075906433423026,\n     'bmi': 0.061696206518683294,\n     'bp': 0.0218723855140367,\n     's1': -0.04422349842444599,\n     's2': -0.03482076283769895,\n     's3': -0.04340084565202491,\n     's4': -0.002592261998183278,\n     's5': 0.019907486170462722,\n     's6': -0.01764612515980379,\n     'sex': 0.05068011873981862}\n    151.0\n\n    "
    kwargs['X'] = dataset.data
    kwargs['y'] = dataset.target
    try:
        kwargs['feature_names'] = dataset.feature_names
    except AttributeError:
        pass
    if isinstance(kwargs['X'], pd.DataFrame):
        yield from stream.iter_pandas(**kwargs)
    else:
        yield from stream.iter_array(**kwargs)