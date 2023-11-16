from abc import abstractmethod
from typing import List
import numpy as np
from scipy.sparse import issparse
from ... import get_config
from .._dist_metrics import BOOL_METRICS, METRIC_MAPPING64, DistanceMetric
from ._argkmin import ArgKmin32, ArgKmin64
from ._argkmin_classmode import ArgKminClassMode32, ArgKminClassMode64
from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64
from ._radius_neighbors import RadiusNeighbors32, RadiusNeighbors64
from ._radius_neighbors_classmode import RadiusNeighborsClassMode32, RadiusNeighborsClassMode64

def sqeuclidean_row_norms(X, num_threads):
    if False:
        return 10
    'Compute the squared euclidean norm of the rows of X in parallel.\n\n    Parameters\n    ----------\n    X : ndarray or CSR matrix of shape (n_samples, n_features)\n        Input data. Must be c-contiguous.\n\n    num_threads : int\n        The number of OpenMP threads to use.\n\n    Returns\n    -------\n    sqeuclidean_row_norms : ndarray of shape (n_samples,)\n        Arrays containing the squared euclidean norm of each row of X.\n    '
    if X.dtype == np.float64:
        return np.asarray(_sqeuclidean_row_norms64(X, num_threads))
    if X.dtype == np.float32:
        return np.asarray(_sqeuclidean_row_norms32(X, num_threads))
    raise ValueError(f'Only float64 or float32 datasets are supported at this time, got: X.dtype={X.dtype}.')

class BaseDistancesReductionDispatcher:
    """Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        if False:
            i = 10
            return i + 15
        excluded = {'pyfunc', 'mahalanobis', 'hamming', *BOOL_METRICS}
        return sorted(({'sqeuclidean'} | set(METRIC_MAPPING64.keys())) - excluded)

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        if False:
            i = 10
            return i + 15
        "Return True if the dispatcher can be used for the\n        given parameters.\n\n        Parameters\n        ----------\n        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)\n            Input data.\n\n        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)\n            Input data.\n\n        metric : str, default='euclidean'\n            The distance metric to use.\n            For a list of available metrics, see the documentation of\n            :class:`~sklearn.metrics.DistanceMetric`.\n\n        Returns\n        -------\n        True if the dispatcher can be used, else False.\n        "

        def is_numpy_c_ordered(X):
            if False:
                for i in range(10):
                    print('nop')
            return hasattr(X, 'flags') and getattr(X.flags, 'c_contiguous', False)

        def is_valid_sparse_matrix(X):
            if False:
                return 10
            return issparse(X) and X.format == 'csr' and (X.nnz > 0) and (X.indices.dtype == X.indptr.dtype == np.int32)
        is_usable = get_config().get('enable_cython_pairwise_dist', True) and (is_numpy_c_ordered(X) or is_valid_sparse_matrix(X)) and (is_numpy_c_ordered(Y) or is_valid_sparse_matrix(Y)) and (X.dtype == Y.dtype) and (X.dtype in (np.float32, np.float64)) and (metric in cls.valid_metrics() or isinstance(metric, DistanceMetric))
        return is_usable

    @classmethod
    @abstractmethod
    def compute(cls, X, Y, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Compute the reduction.\n\n        Parameters\n        ----------\n        X : ndarray or CSR matrix of shape (n_samples_X, n_features)\n            Input data.\n\n        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)\n            Input data.\n\n        **kwargs : additional parameters for the reduction\n\n        Notes\n        -----\n        This method is an abstract class method: it has to be implemented\n        for all subclasses.\n        '

class ArgKmin(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances.

    ArgKmin is typically used to perform
    bruteforce k-nearest neighbors queries.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(cls, X, Y, k, metric='euclidean', chunk_size=None, metric_kwargs=None, strategy=None, return_distance=False):
        if False:
            for i in range(10):
                print('nop')
        "Compute the argkmin reduction.\n\n        Parameters\n        ----------\n        X : ndarray or CSR matrix of shape (n_samples_X, n_features)\n            Input data.\n\n        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)\n            Input data.\n\n        k : int\n            The k for the argkmin reduction.\n\n        metric : str, default='euclidean'\n            The distance metric to use for argkmin.\n            For a list of available metrics, see the documentation of\n            :class:`~sklearn.metrics.DistanceMetric`.\n\n        chunk_size : int, default=None,\n            The number of vectors per chunk. If None (default) looks-up in\n            scikit-learn configuration for `pairwise_dist_chunk_size`,\n            and use 256 if it is not set.\n\n        metric_kwargs : dict, default=None\n            Keyword arguments to pass to specified metric function.\n\n        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None\n            The chunking strategy defining which dataset parallelization are made on.\n\n            For both strategies the computations happens with two nested loops,\n            respectively on chunks of X and chunks of Y.\n            Strategies differs on which loop (outer or inner) is made to run\n            in parallel with the Cython `prange` construct:\n\n              - 'parallel_on_X' dispatches chunks of X uniformly on threads.\n                Each thread then iterates on all the chunks of Y. This strategy is\n                embarrassingly parallel and comes with no datastructures\n                synchronisation.\n\n              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.\n                Each thread processes all the chunks of X in turn. This strategy is\n                a sequence of embarrassingly parallel subtasks (the inner loop on Y\n                chunks) with intermediate datastructures synchronisation at each\n                iteration of the sequential outer loop on X chunks.\n\n              - 'auto' relies on a simple heuristic to choose between\n                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,\n                'parallel_on_X' is usually the most efficient strategy.\n                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'\n                brings more opportunity for parallelism and is therefore more efficient\n\n              - None (default) looks-up in scikit-learn configuration for\n                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.\n\n        return_distance : boolean, default=False\n            Return distances between each X vector and its\n            argkmin if set to True.\n\n        Returns\n        -------\n        If return_distance=False:\n          - argkmin_indices : ndarray of shape (n_samples_X, k)\n            Indices of the argkmin for each vector in X.\n\n        If return_distance=True:\n          - argkmin_distances : ndarray of shape (n_samples_X, k)\n            Distances to the argkmin for each vector in X.\n          - argkmin_indices : ndarray of shape (n_samples_X, k)\n            Indices of the argkmin for each vector in X.\n\n        Notes\n        -----\n        This classmethod inspects the arguments values to dispatch to the\n        dtype-specialized implementation of :class:`ArgKmin`.\n\n        This allows decoupling the API entirely from the implementation details\n        whilst maintaining RAII: all temporarily allocated datastructures necessary\n        for the concrete implementation are therefore freed when this classmethod\n        returns.\n        "
        if X.dtype == Y.dtype == np.float64:
            return ArgKmin64.compute(X=X, Y=Y, k=k, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy, return_distance=return_distance)
        if X.dtype == Y.dtype == np.float32:
            return ArgKmin32.compute(X=X, Y=Y, k=k, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy, return_distance=return_distance)
        raise ValueError(f'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype={X.dtype} and Y.dtype={Y.dtype}.')

class RadiusNeighbors(BaseDistancesReductionDispatcher):
    """Compute radius-based neighbors for two sets of vectors.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    The distance function `dist` depends on the values of the `metric`
    and `metric_kwargs` parameters.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(cls, X, Y, radius, metric='euclidean', chunk_size=None, metric_kwargs=None, strategy=None, return_distance=False, sort_results=False):
        if False:
            for i in range(10):
                print('nop')
        "Return the results of the reduction for the given arguments.\n\n        Parameters\n        ----------\n        X : ndarray or CSR matrix of shape (n_samples_X, n_features)\n            Input data.\n\n        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)\n            Input data.\n\n        radius : float\n            The radius defining the neighborhood.\n\n        metric : str, default='euclidean'\n            The distance metric to use.\n            For a list of available metrics, see the documentation of\n            :class:`~sklearn.metrics.DistanceMetric`.\n\n        chunk_size : int, default=None,\n            The number of vectors per chunk. If None (default) looks-up in\n            scikit-learn configuration for `pairwise_dist_chunk_size`,\n            and use 256 if it is not set.\n\n        metric_kwargs : dict, default=None\n            Keyword arguments to pass to specified metric function.\n\n        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None\n            The chunking strategy defining which dataset parallelization are made on.\n\n            For both strategies the computations happens with two nested loops,\n            respectively on chunks of X and chunks of Y.\n            Strategies differs on which loop (outer or inner) is made to run\n            in parallel with the Cython `prange` construct:\n\n              - 'parallel_on_X' dispatches chunks of X uniformly on threads.\n                Each thread then iterates on all the chunks of Y. This strategy is\n                embarrassingly parallel and comes with no datastructures\n                synchronisation.\n\n              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.\n                Each thread processes all the chunks of X in turn. This strategy is\n                a sequence of embarrassingly parallel subtasks (the inner loop on Y\n                chunks) with intermediate datastructures synchronisation at each\n                iteration of the sequential outer loop on X chunks.\n\n              - 'auto' relies on a simple heuristic to choose between\n                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,\n                'parallel_on_X' is usually the most efficient strategy.\n                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'\n                brings more opportunity for parallelism and is therefore more efficient\n                despite the synchronization step at each iteration of the outer loop\n                on chunks of `X`.\n\n              - None (default) looks-up in scikit-learn configuration for\n                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.\n\n        return_distance : boolean, default=False\n            Return distances between each X vector and its neighbors if set to True.\n\n        sort_results : boolean, default=False\n            Sort results with respect to distances between each X vector and its\n            neighbors if set to True.\n\n        Returns\n        -------\n        If return_distance=False:\n          - neighbors_indices : ndarray of n_samples_X ndarray\n            Indices of the neighbors for each vector in X.\n\n        If return_distance=True:\n          - neighbors_indices : ndarray of n_samples_X ndarray\n            Indices of the neighbors for each vector in X.\n          - neighbors_distances : ndarray of n_samples_X ndarray\n            Distances to the neighbors for each vector in X.\n\n        Notes\n        -----\n        This classmethod inspects the arguments values to dispatch to the\n        dtype-specialized implementation of :class:`RadiusNeighbors`.\n\n        This allows decoupling the API entirely from the implementation details\n        whilst maintaining RAII: all temporarily allocated datastructures necessary\n        for the concrete implementation are therefore freed when this classmethod\n        returns.\n        "
        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighbors64.compute(X=X, Y=Y, radius=radius, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy, sort_results=sort_results, return_distance=return_distance)
        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighbors32.compute(X=X, Y=Y, radius=radius, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy, sort_results=sort_results, return_distance=return_distance)
        raise ValueError(f'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype={X.dtype} and Y.dtype={Y.dtype}.')

class ArgKminClassMode(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y with labels.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances. Computes weighted mode of labels.

    ArgKminClassMode is typically used to perform bruteforce k-nearest neighbors
    queries when the weighted mode of the labels for the k-nearest neighbors
    are required, such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        excluded = {'euclidean', 'sqeuclidean'}
        return list(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(cls, X, Y, k, weights, Y_labels, unique_Y_labels, metric='euclidean', chunk_size=None, metric_kwargs=None, strategy=None):
        if False:
            while True:
                i = 10
        "Compute the argkmin reduction.\n\n        Parameters\n        ----------\n        X : ndarray of shape (n_samples_X, n_features)\n            The input array to be labelled.\n\n        Y : ndarray of shape (n_samples_Y, n_features)\n            The input array whose class membership are provided through the\n            `Y_labels` parameter.\n\n        k : int\n            The number of nearest neighbors to consider.\n\n        weights : ndarray\n            The weights applied over the `Y_labels` of `Y` when computing the\n            weighted mode of the labels.\n\n        Y_labels : ndarray\n            An array containing the index of the class membership of the\n            associated samples in `Y`. This is used in labeling `X`.\n\n        unique_Y_labels : ndarray\n            An array containing all unique indices contained in the\n            corresponding `Y_labels` array.\n\n        metric : str, default='euclidean'\n            The distance metric to use. For a list of available metrics, see\n            the documentation of :class:`~sklearn.metrics.DistanceMetric`.\n            Currently does not support `'precomputed'`.\n\n        chunk_size : int, default=None,\n            The number of vectors per chunk. If None (default) looks-up in\n            scikit-learn configuration for `pairwise_dist_chunk_size`,\n            and use 256 if it is not set.\n\n        metric_kwargs : dict, default=None\n            Keyword arguments to pass to specified metric function.\n\n        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None\n            The chunking strategy defining which dataset parallelization are made on.\n\n            For both strategies the computations happens with two nested loops,\n            respectively on chunks of X and chunks of Y.\n            Strategies differs on which loop (outer or inner) is made to run\n            in parallel with the Cython `prange` construct:\n\n              - 'parallel_on_X' dispatches chunks of X uniformly on threads.\n                Each thread then iterates on all the chunks of Y. This strategy is\n                embarrassingly parallel and comes with no datastructures\n                synchronisation.\n\n              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.\n                Each thread processes all the chunks of X in turn. This strategy is\n                a sequence of embarrassingly parallel subtasks (the inner loop on Y\n                chunks) with intermediate datastructures synchronisation at each\n                iteration of the sequential outer loop on X chunks.\n\n              - 'auto' relies on a simple heuristic to choose between\n                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,\n                'parallel_on_X' is usually the most efficient strategy.\n                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'\n                brings more opportunity for parallelism and is therefore more efficient\n                despite the synchronization step at each iteration of the outer loop\n                on chunks of `X`.\n\n              - None (default) looks-up in scikit-learn configuration for\n                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.\n\n        Returns\n        -------\n        probabilities : ndarray of shape (n_samples_X, n_classes)\n            An array containing the class probabilities for each sample.\n\n        Notes\n        -----\n        This classmethod is responsible for introspecting the arguments\n        values to dispatch to the most appropriate implementation of\n        :class:`PairwiseDistancesArgKmin`.\n\n        This allows decoupling the API entirely from the implementation details\n        whilst maintaining RAII: all temporarily allocated datastructures necessary\n        for the concrete implementation are therefore freed when this classmethod\n        returns.\n        "
        if weights not in {'uniform', 'distance'}:
            raise ValueError(f"Only the 'uniform' or 'distance' weights options are supported at this time. Got: weights={weights!r}.")
        if X.dtype == Y.dtype == np.float64:
            return ArgKminClassMode64.compute(X=X, Y=Y, k=k, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        if X.dtype == Y.dtype == np.float32:
            return ArgKminClassMode32.compute(X=X, Y=Y, k=k, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        raise ValueError(f'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype={X.dtype} and Y.dtype={Y.dtype}.')

class RadiusNeighborsClassMode(BaseDistancesReductionDispatcher):
    """Compute radius-based class modes of row vectors of X using the
    those of Y.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    RadiusNeighborsClassMode is typically used to perform bruteforce
    radius neighbors queries when the weighted mode of the labels for
    the nearest neighbors within the specified radius are required,
    such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        if False:
            return 10
        excluded = {'euclidean', 'sqeuclidean'}
        return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(cls, X, Y, radius, weights, Y_labels, unique_Y_labels, outlier_label, metric='euclidean', chunk_size=None, metric_kwargs=None, strategy=None):
        if False:
            while True:
                i = 10
        "Return the results of the reduction for the given arguments.\n        Parameters\n        ----------\n        X : ndarray of shape (n_samples_X, n_features)\n            The input array to be labelled.\n        Y : ndarray of shape (n_samples_Y, n_features)\n            The input array whose class membership is provided through\n            the `Y_labels` parameter.\n        radius : float\n            The radius defining the neighborhood.\n        weights : ndarray\n            The weights applied to the `Y_labels` when computing the\n            weighted mode of the labels.\n        Y_labels : ndarray\n            An array containing the index of the class membership of the\n            associated samples in `Y`. This is used in labeling `X`.\n        unique_Y_labels : ndarray\n            An array containing all unique class labels.\n        outlier_label : int, default=None\n            Label for outlier samples (samples with no neighbors in given\n            radius). In the default case when the value is None if any\n            outlier is detected, a ValueError will be raised. The outlier\n            label should be selected from among the unique 'Y' labels. If\n            it is specified with a different value a warning will be raised\n            and all class probabilities of outliers will be assigned to be 0.\n        metric : str, default='euclidean'\n            The distance metric to use. For a list of available metrics, see\n            the documentation of :class:`~sklearn.metrics.DistanceMetric`.\n            Currently does not support `'precomputed'`.\n        chunk_size : int, default=None,\n            The number of vectors per chunk. If None (default) looks-up in\n            scikit-learn configuration for `pairwise_dist_chunk_size`,\n            and use 256 if it is not set.\n        metric_kwargs : dict, default=None\n            Keyword arguments to pass to specified metric function.\n        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None\n            The chunking strategy defining which dataset parallelization are made on.\n            For both strategies the computations happens with two nested loops,\n            respectively on chunks of X and chunks of Y.\n            Strategies differs on which loop (outer or inner) is made to run\n            in parallel with the Cython `prange` construct:\n              - 'parallel_on_X' dispatches chunks of X uniformly on threads.\n                Each thread then iterates on all the chunks of Y. This strategy is\n                embarrassingly parallel and comes with no datastructures\n                synchronisation.\n              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.\n                Each thread processes all the chunks of X in turn. This strategy is\n                a sequence of embarrassingly parallel subtasks (the inner loop on Y\n                chunks) with intermediate datastructures synchronisation at each\n                iteration of the sequential outer loop on X chunks.\n              - 'auto' relies on a simple heuristic to choose between\n                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,\n                'parallel_on_X' is usually the most efficient strategy.\n                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'\n                brings more opportunity for parallelism and is therefore more efficient\n                despite the synchronization step at each iteration of the outer loop\n                on chunks of `X`.\n              - None (default) looks-up in scikit-learn configuration for\n                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.\n        Returns\n        -------\n        probabilities : ndarray of shape (n_samples_X, n_classes)\n            An array containing the class probabilities for each sample.\n        "
        if weights not in {'uniform', 'distance'}:
            raise ValueError(f"Only the 'uniform' or 'distance' weights options are supported at this time. Got: weights={weights!r}.")
        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighborsClassMode64.compute(X=X, Y=Y, radius=radius, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), outlier_label=outlier_label, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighborsClassMode32.compute(X=X, Y=Y, radius=radius, weights=weights, Y_labels=np.array(Y_labels, dtype=np.intp), unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp), outlier_label=outlier_label, metric=metric, chunk_size=chunk_size, metric_kwargs=metric_kwargs, strategy=strategy)
        raise ValueError(f'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype={X.dtype} and Y.dtype={Y.dtype}.')