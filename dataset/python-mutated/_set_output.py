from functools import wraps
from scipy.sparse import issparse
from .._config import get_config
from . import check_pandas_support
from ._available_if import available_if
from .validation import _is_pandas_df

def _wrap_in_pandas_container(data_to_wrap, *, columns, index=None):
    if False:
        return 10
    'Create a Pandas DataFrame.\n\n    If `data_to_wrap` is a DataFrame, then the `columns` and `index` will be changed\n    inplace. If `data_to_wrap` is a ndarray, then a new DataFrame is created with\n    `columns` and `index`.\n\n    Parameters\n    ----------\n    data_to_wrap : {ndarray, dataframe}\n        Data to be wrapped as pandas dataframe.\n\n    columns : callable, ndarray, or None\n        The column names or a callable that returns the column names. The\n        callable is useful if the column names require some computation.\n        If `columns` is a callable that raises an error, `columns` will have\n        the same semantics as `None`. If `None` and `data_to_wrap` is already a\n        dataframe, then the column names are not changed. If `None` and\n        `data_to_wrap` is **not** a dataframe, then columns are\n        `range(n_features)`.\n\n    index : array-like, default=None\n        Index for data. `index` is ignored if `data_to_wrap` is already a DataFrame.\n\n    Returns\n    -------\n    dataframe : DataFrame\n        Container with column names or unchanged `output`.\n    '
    if issparse(data_to_wrap):
        raise ValueError("The transformer outputs a scipy sparse matrix. Try to set the transformer output to a dense array or disable pandas output with set_output(transform='default').")
    if callable(columns):
        try:
            columns = columns()
        except Exception:
            columns = None
    pd = check_pandas_support("Setting output container to 'pandas'")
    if isinstance(data_to_wrap, pd.DataFrame):
        if columns is not None:
            data_to_wrap.columns = columns
        return data_to_wrap
    return pd.DataFrame(data_to_wrap, index=index, columns=columns, copy=False)

def _get_output_config(method, estimator=None):
    if False:
        i = 10
        return i + 15
    'Get output config based on estimator and global configuration.\n\n    Parameters\n    ----------\n    method : {"transform"}\n        Estimator\'s method for which the output container is looked up.\n\n    estimator : estimator instance or None\n        Estimator to get the output configuration from. If `None`, check global\n        configuration is used.\n\n    Returns\n    -------\n    config : dict\n        Dictionary with keys:\n\n        - "dense": specifies the dense container for `method`. This can be\n          `"default"` or `"pandas"`.\n    '
    est_sklearn_output_config = getattr(estimator, '_sklearn_output_config', {})
    if method in est_sklearn_output_config:
        dense_config = est_sklearn_output_config[method]
    else:
        dense_config = get_config()[f'{method}_output']
    if dense_config not in {'default', 'pandas'}:
        raise ValueError(f"output config must be 'default' or 'pandas' got {dense_config}")
    return {'dense': dense_config}

def _wrap_data_with_container(method, data_to_wrap, original_input, estimator):
    if False:
        i = 10
        return i + 15
    'Wrap output with container based on an estimator\'s or global config.\n\n    Parameters\n    ----------\n    method : {"transform"}\n        Estimator\'s method to get container output for.\n\n    data_to_wrap : {ndarray, dataframe}\n        Data to wrap with container.\n\n    original_input : {ndarray, dataframe}\n        Original input of function.\n\n    estimator : estimator instance\n        Estimator with to get the output configuration from.\n\n    Returns\n    -------\n    output : {ndarray, dataframe}\n        If the output config is "default" or the estimator is not configured\n        for wrapping return `data_to_wrap` unchanged.\n        If the output config is "pandas", return `data_to_wrap` as a pandas\n        DataFrame.\n    '
    output_config = _get_output_config(method, estimator)
    if output_config['dense'] == 'default' or not _auto_wrap_is_configured(estimator):
        return data_to_wrap
    index = original_input.index if _is_pandas_df(original_input) else None
    return _wrap_in_pandas_container(data_to_wrap=data_to_wrap, index=index, columns=estimator.get_feature_names_out)

def _wrap_method_output(f, method):
    if False:
        i = 10
        return i + 15
    'Wrapper used by `_SetOutputMixin` to automatically wrap methods.'

    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        data_to_wrap = f(self, X, *args, **kwargs)
        if isinstance(data_to_wrap, tuple):
            return_tuple = (_wrap_data_with_container(method, data_to_wrap[0], X, self), *data_to_wrap[1:])
            if hasattr(type(data_to_wrap), '_make'):
                return type(data_to_wrap)._make(return_tuple)
            return return_tuple
        return _wrap_data_with_container(method, data_to_wrap, X, self)
    return wrapped

def _auto_wrap_is_configured(estimator):
    if False:
        while True:
            i = 10
    'Return True if estimator is configured for auto-wrapping the transform method.\n\n    `_SetOutputMixin` sets `_sklearn_auto_wrap_output_keys` to `set()` if auto wrapping\n    is manually disabled.\n    '
    auto_wrap_output_keys = getattr(estimator, '_sklearn_auto_wrap_output_keys', set())
    return hasattr(estimator, 'get_feature_names_out') and 'transform' in auto_wrap_output_keys

class _SetOutputMixin:
    """Mixin that dynamically wraps methods to return container based on config.

    Currently `_SetOutputMixin` wraps `transform` and `fit_transform` and configures
    it based on `set_output` of the global configuration.

    `set_output` is only defined if `get_feature_names_out` is defined and
    `auto_wrap_output_keys` is the default value.
    """

    def __init_subclass__(cls, auto_wrap_output_keys=('transform',), **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init_subclass__(**kwargs)
        if not (isinstance(auto_wrap_output_keys, tuple) or auto_wrap_output_keys is None):
            raise ValueError('auto_wrap_output_keys must be None or a tuple of keys.')
        if auto_wrap_output_keys is None:
            cls._sklearn_auto_wrap_output_keys = set()
            return
        method_to_key = {'transform': 'transform', 'fit_transform': 'transform'}
        cls._sklearn_auto_wrap_output_keys = set()
        for (method, key) in method_to_key.items():
            if not hasattr(cls, method) or key not in auto_wrap_output_keys:
                continue
            cls._sklearn_auto_wrap_output_keys.add(key)
            if method not in cls.__dict__:
                continue
            wrapped_method = _wrap_method_output(getattr(cls, method), key)
            setattr(cls, method, wrapped_method)

    @available_if(_auto_wrap_is_configured)
    def set_output(self, *, transform=None):
        if False:
            for i in range(10):
                print('nop')
        'Set output container.\n\n        See :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`\n        for an example on how to use the API.\n\n        Parameters\n        ----------\n        transform : {"default", "pandas"}, default=None\n            Configure output of `transform` and `fit_transform`.\n\n            - `"default"`: Default output format of a transformer\n            - `"pandas"`: DataFrame output\n            - `None`: Transform configuration is unchanged\n\n        Returns\n        -------\n        self : estimator instance\n            Estimator instance.\n        '
        if transform is None:
            return self
        if not hasattr(self, '_sklearn_output_config'):
            self._sklearn_output_config = {}
        self._sklearn_output_config['transform'] = transform
        return self

def _safe_set_output(estimator, *, transform=None):
    if False:
        while True:
            i = 10
    'Safely call estimator.set_output and error if it not available.\n\n    This is used by meta-estimators to set the output for child estimators.\n\n    Parameters\n    ----------\n    estimator : estimator instance\n        Estimator instance.\n\n    transform : {"default", "pandas"}, default=None\n        Configure output of the following estimator\'s methods:\n\n        - `"transform"`\n        - `"fit_transform"`\n\n        If `None`, this operation is a no-op.\n\n    Returns\n    -------\n    estimator : estimator instance\n        Estimator instance.\n    '
    set_output_for_transform = hasattr(estimator, 'transform') or (hasattr(estimator, 'fit_transform') and transform is not None)
    if not set_output_for_transform:
        return
    if not hasattr(estimator, 'set_output'):
        raise ValueError(f'Unable to configure output for {estimator} because `set_output` is not available.')
    return estimator.set_output(transform=transform)