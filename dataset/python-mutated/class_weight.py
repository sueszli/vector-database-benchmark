"""
The :mod:`sklearn.utils.class_weight` module includes utilities for handling
weights based on class labels.
"""
import numpy as np
from scipy import sparse
from ._param_validation import StrOptions, validate_params

@validate_params({'class_weight': [dict, StrOptions({'balanced'}), None], 'classes': [np.ndarray], 'y': ['array-like']}, prefer_skip_nested_validation=True)
def compute_class_weight(class_weight, *, classes, y):
    if False:
        for i in range(10):
            print('nop')
    'Estimate class weights for unbalanced datasets.\n\n    Parameters\n    ----------\n    class_weight : dict, "balanced" or None\n        If "balanced", class weights will be given by\n        `n_samples / (n_classes * np.bincount(y))`.\n        If a dictionary is given, keys are classes and values are corresponding class\n        weights.\n        If `None` is given, the class weights will be uniform.\n\n    classes : ndarray\n        Array of the classes occurring in the data, as given by\n        `np.unique(y_org)` with `y_org` the original class labels.\n\n    y : array-like of shape (n_samples,)\n        Array of original class labels per sample.\n\n    Returns\n    -------\n    class_weight_vect : ndarray of shape (n_classes,)\n        Array with `class_weight_vect[i]` the weight for i-th class.\n\n    References\n    ----------\n    The "balanced" heuristic is inspired by\n    Logistic Regression in Rare Events Data, King, Zen, 2001.\n    '
    from ..preprocessing import LabelEncoder
    if set(y) - set(classes):
        raise ValueError('classes should include all valid labels that can be in y')
    if class_weight is None or len(class_weight) == 0:
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all(np.isin(classes, le.classes_)):
            raise ValueError('classes should have valid labels that are in y')
        recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        unweighted_classes = []
        for (i, c) in enumerate(classes):
            if c in class_weight:
                weight[i] = class_weight[c]
            else:
                unweighted_classes.append(c)
        n_weighted_classes = len(classes) - len(unweighted_classes)
        if unweighted_classes and n_weighted_classes != len(class_weight):
            unweighted_classes_user_friendly_str = np.array(unweighted_classes).tolist()
            raise ValueError(f'The classes, {unweighted_classes_user_friendly_str}, are not in class_weight')
    return weight

@validate_params({'class_weight': [dict, list, StrOptions({'balanced'}), None], 'y': ['array-like', 'sparse matrix'], 'indices': ['array-like', None]}, prefer_skip_nested_validation=True)
def compute_sample_weight(class_weight, y, *, indices=None):
    if False:
        while True:
            i = 10
    'Estimate sample weights by class for unbalanced datasets.\n\n    Parameters\n    ----------\n    class_weight : dict, list of dicts, "balanced", or None\n        Weights associated with classes in the form `{class_label: weight}`.\n        If not given, all classes are supposed to have weight one. For\n        multi-output problems, a list of dicts can be provided in the same\n        order as the columns of y.\n\n        Note that for multioutput (including multilabel) weights should be\n        defined for each class of every column in its own dict. For example,\n        for four-class multilabel classification weights should be\n        `[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]` instead of\n        `[{1:1}, {2:5}, {3:1}, {4:1}]`.\n\n        The `"balanced"` mode uses the values of y to automatically adjust\n        weights inversely proportional to class frequencies in the input data:\n        `n_samples / (n_classes * np.bincount(y))`.\n\n        For multi-output, the weights of each column of y will be multiplied.\n\n    y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)\n        Array of original class labels per sample.\n\n    indices : array-like of shape (n_subsample,), default=None\n        Array of indices to be used in a subsample. Can be of length less than\n        `n_samples` in the case of a subsample, or equal to `n_samples` in the\n        case of a bootstrap subsample with repeated indices. If `None`, the\n        sample weight will be calculated over the full sample. Only `"balanced"`\n        is supported for `class_weight` if this is provided.\n\n    Returns\n    -------\n    sample_weight_vect : ndarray of shape (n_samples,)\n        Array with sample weights as applied to the original `y`.\n    '
    if not sparse.issparse(y):
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
    n_outputs = y.shape[1]
    if indices is not None and class_weight != 'balanced':
        raise ValueError(f"The only valid class_weight for subsampling is 'balanced'. Given {class_weight}.")
    elif n_outputs > 1:
        if class_weight is None or isinstance(class_weight, dict):
            raise ValueError("For multi-output, class_weight should be a list of dicts, or the string 'balanced'.")
        elif isinstance(class_weight, list) and len(class_weight) != n_outputs:
            raise ValueError(f'For multi-output, number of elements in class_weight should match number of outputs. Got {len(class_weight)} element(s) while having {n_outputs} outputs.')
    expanded_class_weight = []
    for k in range(n_outputs):
        if sparse.issparse(y):
            y_full = y[:, [k]].toarray().flatten()
        else:
            y_full = y[:, k]
        classes_full = np.unique(y_full)
        classes_missing = None
        if class_weight == 'balanced' or n_outputs == 1:
            class_weight_k = class_weight
        else:
            class_weight_k = class_weight[k]
        if indices is not None:
            y_subsample = y_full[indices]
            classes_subsample = np.unique(y_subsample)
            weight_k = np.take(compute_class_weight(class_weight_k, classes=classes_subsample, y=y_subsample), np.searchsorted(classes_subsample, classes_full), mode='clip')
            classes_missing = set(classes_full) - set(classes_subsample)
        else:
            weight_k = compute_class_weight(class_weight_k, classes=classes_full, y=y_full)
        weight_k = weight_k[np.searchsorted(classes_full, y_full)]
        if classes_missing:
            weight_k[np.isin(y_full, list(classes_missing))] = 0.0
        expanded_class_weight.append(weight_k)
    expanded_class_weight = np.prod(expanded_class_weight, axis=0, dtype=np.float64)
    return expanded_class_weight