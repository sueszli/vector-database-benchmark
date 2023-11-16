"""
The :mod:`sklearn.utils.metaestimators` module includes utilities for meta-estimators.
"""
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import Any, List
import numpy as np
from ..base import BaseEstimator
from ..utils import _safe_indexing
from ..utils._tags import _safe_tags
from ._available_if import available_if
__all__ = ['available_if']

class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for classifiers composed of named estimators."""
    steps: List[Any]

    @abstractmethod
    def __init__(self):
        if False:
            return 10
        pass

    def _get_params(self, attr, deep=True):
        if False:
            print('Hello World!')
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            return out
        for (name, estimator) in estimators:
            if hasattr(estimator, 'get_params'):
                for (key, value) in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        if False:
            return 10
        if attr in params:
            setattr(self, attr, params.pop(attr))
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            with suppress(TypeError):
                (item_names, _) = zip(*items)
                for name in list(params.keys()):
                    if '__' not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        if False:
            return 10
        new_estimators = list(getattr(self, attr))
        for (i, (estimator_name, _)) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names):
        if False:
            while True:
                i = 10
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: {0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got {0!r}'.format(invalid_names))

def _safe_split(estimator, X, y, indices, train_indices=None):
    if False:
        for i in range(10):
            print('nop')
    'Create subset of dataset and properly handle kernels.\n\n    Slice X, y according to indices for cross-validation, but take care of\n    precomputed kernel-matrices or pairwise affinities / distances.\n\n    If ``estimator._pairwise is True``, X needs to be square and\n    we slice rows and columns. If ``train_indices`` is not None,\n    we slice rows using ``indices`` (assumed the test set) and columns\n    using ``train_indices``, indicating the training set.\n\n    Labels y will always be indexed only along the first axis.\n\n    Parameters\n    ----------\n    estimator : object\n        Estimator to determine whether we should slice only rows or rows and\n        columns.\n\n    X : array-like, sparse matrix or iterable\n        Data to be indexed. If ``estimator._pairwise is True``,\n        this needs to be a square array-like or sparse matrix.\n\n    y : array-like, sparse matrix or iterable\n        Targets to be indexed.\n\n    indices : array of int\n        Rows to select from X and y.\n        If ``estimator._pairwise is True`` and ``train_indices is None``\n        then ``indices`` will also be used to slice columns.\n\n    train_indices : array of int or None, default=None\n        If ``estimator._pairwise is True`` and ``train_indices is not None``,\n        then ``train_indices`` will be use to slice the columns of X.\n\n    Returns\n    -------\n    X_subset : array-like, sparse matrix or list\n        Indexed data.\n\n    y_subset : array-like, sparse matrix or list\n        Indexed targets.\n\n    '
    if _safe_tags(estimator, key='pairwise'):
        if not hasattr(X, 'shape'):
            raise ValueError('Precomputed kernels or affinity matrices have to be passed as arrays or sparse matrices.')
        if X.shape[0] != X.shape[1]:
            raise ValueError('X should be a square kernel matrix')
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)
    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None
    return (X_subset, y_subset)