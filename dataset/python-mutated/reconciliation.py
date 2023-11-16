"""
Hierarchical Reconciliation
---------------------------

A set of posthoc hierarchical reconciliation transformers. These transformers
work on any ``TimeSeries`` (e.g., a forecast) that contain a ``hierarchy``.

A ``hierarchy`` is a dict that maps each component to their parent(s) in the hierarchy.
It can be added to a ``TimeSeries`` using e.g., the :meth:`TimeSeries.with_hierarchy` method.
"""
from typing import Any, Mapping, Optional
import numpy as np
from darts.dataprocessing.transformers import BaseDataTransformer, FittableDataTransformer
from darts.timeseries import TimeSeries
from darts.utils.utils import raise_if_not

def _get_summation_matrix(series: TimeSeries):
    if False:
        return 10
    '\n    Returns the matrix S for a series, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.\n\n    The dimension of the matrix is `(n, m)`, where `n` is the number of components and `m` the number\n    of base components (components that are not the sum of any other components).\n    S[i, j] contains 1 if component i is "made up" of base component j, and 0 otherwise.\n    The order of the `n` and `m` components in the matrix match the order of the components in the `series`.\n\n    The matrix is built using the ``hierarchy`` property of the ``series``. ``hierarchy`` must be a\n    dictionary mapping each (non top-level) component to its parent(s) in the aggregation.\n    '
    raise_if_not(series.has_hierarchy, 'The provided series must have a hierarchy defined for reconciliation to be performed.')
    hierarchy = series.hierarchy
    components_seq = list(series.components)
    leaves_seq = series.bottom_level_components
    m = len(leaves_seq)
    n = len(components_seq)
    S = np.zeros((n, m))
    components_indexes = {c: i for (i, c) in enumerate(components_seq)}
    leaves_indexes = {l: i for (i, l) in enumerate(leaves_seq)}

    def increment(cur_node, leaf_idx):
        if False:
            i = 10
            return i + 15
        '\n        Recursive function filling S for a given base component and all its ancestors\n        '
        S[components_indexes[cur_node], leaf_idx] = 1.0
        if cur_node in hierarchy:
            for parent in hierarchy[cur_node]:
                increment(parent, leaf_idx)
    for leaf in leaves_seq:
        leaf_idx = leaves_indexes[leaf]
        increment(leaf, leaf_idx)
    return S.astype(series.dtype)

def _reconcile_from_S_and_G(series: TimeSeries, S: np.ndarray, G: np.ndarray) -> TimeSeries:
    if False:
        i = 10
        return i + 15
    '\n    Returns the TimeSeries linearly reconciled from the projection matrix G and the summation matrix S.\n    '
    y_hat = series.all_values(copy=False)
    reconciled_values = S @ G @ y_hat
    return series.with_values(reconciled_values)

class BottomUpReconciliator(BaseDataTransformer):
    """
    Performs bottom up reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.
    """

    @staticmethod
    def get_projection_matrix(series):
        if False:
            i = 10
            return i + 15
        leaves_seq = list(series.bottom_level_components)
        (n, m) = (series.n_components, len(leaves_seq))
        leaves_indexes = {l: i for (i, l) in enumerate(leaves_seq)}
        G = np.zeros((m, n))
        for (i, c) in enumerate(series.components):
            if c in leaves_indexes:
                G[leaves_indexes[c], i] = 1.0
        return G

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], *args, **kwargs) -> TimeSeries:
        if False:
            return 10
        S = _get_summation_matrix(series)
        G = BottomUpReconciliator.get_projection_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)

class TopDownReconciliator(FittableDataTransformer):
    """
    Performs top down reconciliation, as defined `here <https://otexts.com/fpp3/reconciliation.html>`_.

    This estimator computes the proportions (of the base components w.r.t. the top component)
    based on the TimeSeries provided to the method :meth:`fit()`. If the historical series
    is provided, then the historical proportions will be used.
    """

    @staticmethod
    def ts_fit(series: TimeSeries, params: Mapping[str, Any], *args, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        G = TopDownReconciliator.get_projection_matrix(series)
        return G

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], *args, **kwargs) -> TimeSeries:
        if False:
            print('Hello World!')
        G = params['fitted']
        S = _get_summation_matrix(series)
        return _reconcile_from_S_and_G(series, S, G)

    @staticmethod
    def get_projection_matrix(series):
        if False:
            print('Hello World!')
        (n, m) = (series.n_components, len(series.bottom_level_components))
        sum_total = series[series.top_level_component].all_values(copy=False).flatten().sum()
        base_forecasts = series.bottom_level_series
        sum_base = base_forecasts.all_values(copy=False).sum(axis=2).sum(axis=0)
        proportions = sum_base / sum_total
        top_level_index = list(series.components).index(series.top_level_component)
        G = np.zeros((m, n))
        G[:, top_level_index] = proportions
        return G

class MinTReconciliator(FittableDataTransformer):

    def __init__(self, method='ols'):
        if False:
            print('Hello World!')
        '\n        MinT Reconcilator.\n\n        This implements the MinT reconciliation approach presented in [1]_ and\n        summarised in [2]_.\n\n        Parameters\n        ----------\n        method\n            This parameter can take four different values, determining how the covariance\n            matrix ``W`` of the forecast errors is estimated (corresponding to ``Wh`` in [2]_):\n\n            * ``ols`` uses ``W = I``. This option looks only at the hierarchy but ignores the\n              values of the series provided to ``fit()``.\n            * ``wls_struct`` uses ``W = diag(S1)``, where ``S1`` is a vector of size `n` with values\n              between 0 and `m`, representing the number of base components composing each\n              of the `n` components. This options looks only at the hierarchy but ignores\n              the values of the series provided to ``fit()``.\n            * ``wls_var`` uses ``W = diag(W1)``, where ``W1`` is the temporal average of the\n              variance of the forecasting residuals. This method assumes that the series\n              provided to ``fit()`` contain the forecast residuals (deterministic series).\n            * ``mint_cov`` computes ``W`` as the empirical covariance matrix of the residuals\n              for each component, with residuals samples taken over time. This method assumes\n              that the series provided to ``fit()`` contain the forecast residuals\n              (deterministic series), and it requires the residuals to be linearly independent.\n            * ``wls_val`` uses ``W = diag(V1)``, where ``V1`` is the temporal average of the\n              component values. This method assumes that the series provided to ``fit()`` contains\n              an example of the actual values (e.g., either the training series or the forecasts).\n              This method is not presented in [2]_.\n\n        References\n        ----------\n        .. [1] `Optimal forecast reconciliation for hierarchical and grouped time series through\n                trace minimization <https://robjhyndman.com/papers/MinT.pdf>`_\n        .. [2] https://otexts.com/fpp3/reconciliation.html#the-mint-optimal-reconciliation-approach\n        '
        known_methods = ['ols', 'wls', 'wls_var', 'wls_struct', 'wls_val', 'mint_cov']
        raise_if_not(method in known_methods, f'The method must be one of {known_methods}')
        self.method = method
        super().__init__()

    @staticmethod
    def ts_fit(series: TimeSeries, params: Mapping[str, Any], *args, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        method = params['fixed']['method']
        (S, G) = MinTReconciliator.get_matrices(series, method)
        return (S, G)

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any], *args, **kwargs) -> TimeSeries:
        if False:
            i = 10
            return i + 15
        (S, G) = params['fitted']
        return _reconcile_from_S_and_G(series, S, G)

    @staticmethod
    def _assert_deterministic(series: TimeSeries):
        if False:
            return 10
        raise_if_not(series.is_deterministic, 'When used with method wls_var or mint_cov, the MinT reconciliator ' + 'has to be fit on a deterministic series ' + 'containing residuals. This series is stochastic.')

    @staticmethod
    def get_matrices(series: Optional[TimeSeries], method: str):
        if False:
            i = 10
            return i + 15
        'Returns the G matrix given a specified reconciliation method.'
        S = _get_summation_matrix(series)
        if method == 'ols':
            G = np.linalg.inv(S.T @ S) @ S.T
            return (S, G)
        elif method == 'wls_struct':
            Wh = np.diag(np.sum(S, axis=1))
        elif method == 'wls_var':
            MinTReconciliator._assert_deterministic(series)
            et2 = series.values(copy=False) ** 2
            Wh = np.diag(et2.mean(axis=0))
        elif method == 'wls_val':
            quantities = series.all_values(copy=False).mean(axis=2).mean(axis=0)
            Wh = np.diag(np.array(quantities))
        elif method == 'mint_cov':
            MinTReconciliator._assert_deterministic(series)
            Wh = np.cov(series.values(copy=False).T)
        else:
            raise_if_not(False, f'Unknown method: {method}')
        Wh_inv = np.linalg.inv(Wh)
        G = np.linalg.inv(S.T @ Wh_inv @ S) @ S.T @ Wh_inv
        return (S, G)