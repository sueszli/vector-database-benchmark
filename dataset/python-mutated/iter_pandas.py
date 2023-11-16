from __future__ import annotations
import pandas as pd
from river import base, stream

def iter_pandas(X: pd.DataFrame, y: pd.Series | pd.DataFrame | None=None, **kwargs) -> base.typing.Stream:
    if False:
        while True:
            i = 10
    "Iterates over the rows of a `pandas.DataFrame`.\n\n    Parameters\n    ----------\n    X\n        A dataframe of features.\n    y\n        A series or a dataframe with one column per target.\n    kwargs\n        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.\n\n    Examples\n    --------\n\n    >>> import pandas as pd\n    >>> from river import stream\n\n    >>> X = pd.DataFrame({\n    ...     'x1': [1, 2, 3, 4],\n    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],\n    ...     'y': [True, False, False, True]\n    ... })\n    >>> y = X.pop('y')\n\n    >>> for xi, yi in stream.iter_pandas(X, y):\n    ...     print(xi, yi)\n    {'x1': 1, 'x2': 'blue'} True\n    {'x1': 2, 'x2': 'yellow'} False\n    {'x1': 3, 'x2': 'yellow'} False\n    {'x1': 4, 'x2': 'blue'} True\n\n    "
    kwargs['feature_names'] = X.columns
    if isinstance(y, pd.DataFrame):
        kwargs['target_names'] = y.columns
    yield from stream.iter_array(X=X.to_numpy(), y=y if y is None else y.to_numpy(), **kwargs)