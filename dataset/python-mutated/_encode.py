from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan

def _unique(values, *, return_inverse=False, return_counts=False):
    if False:
        i = 10
        return i + 15
    'Helper function to find unique values with support for python objects.\n\n    Uses pure python method for object dtype, and numpy method for\n    all other dtypes.\n\n    Parameters\n    ----------\n    values : ndarray\n        Values to check for unknowns.\n\n    return_inverse : bool, default=False\n        If True, also return the indices of the unique values.\n\n    return_counts : bool, default=False\n        If True, also return the number of times each unique item appears in\n        values.\n\n    Returns\n    -------\n    unique : ndarray\n        The sorted unique values.\n\n    unique_inverse : ndarray\n        The indices to reconstruct the original array from the unique array.\n        Only provided if `return_inverse` is True.\n\n    unique_counts : ndarray\n        The number of times each of the unique values comes up in the original\n        array. Only provided if `return_counts` is True.\n    '
    if values.dtype == object:
        return _unique_python(values, return_inverse=return_inverse, return_counts=return_counts)
    return _unique_np(values, return_inverse=return_inverse, return_counts=return_counts)

def _unique_np(values, return_inverse=False, return_counts=False):
    if False:
        return 10
    'Helper function to find unique values for numpy arrays that correctly\n    accounts for nans. See `_unique` documentation for details.'
    uniques = np.unique(values, return_inverse=return_inverse, return_counts=return_counts)
    (inverse, counts) = (None, None)
    if return_counts:
        (*uniques, counts) = uniques
    if return_inverse:
        (*uniques, inverse) = uniques
    if return_counts or return_inverse:
        uniques = uniques[0]
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = np.searchsorted(uniques, np.nan)
        uniques = uniques[:nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx
        if return_counts:
            counts[nan_idx] = np.sum(counts[nan_idx:])
            counts = counts[:nan_idx + 1]
    ret = (uniques,)
    if return_inverse:
        ret += (inverse,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret

class MissingValues(NamedTuple):
    """Data class for missing data information"""
    nan: bool
    none: bool

    def to_list(self):
        if False:
            return 10
        'Convert tuple to a list where None is always first.'
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output

def _extract_missing(values):
    if False:
        while True:
            i = 10
    'Extract missing values from `values`.\n\n    Parameters\n    ----------\n    values: set\n        Set of values to extract missing from.\n\n    Returns\n    -------\n    output: set\n        Set with missing values extracted.\n\n    missing_values: MissingValues\n        Object with missing value information.\n    '
    missing_values_set = {value for value in values if value is None or is_scalar_nan(value)}
    if not missing_values_set:
        return (values, MissingValues(nan=False, none=False))
    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)
    output = values - missing_values_set
    return (output, output_missing_values)

class _nandict(dict):
    """Dictionary with support for nans."""

    def __init__(self, mapping):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(mapping)
        for (key, value) in mapping.items():
            if is_scalar_nan(key):
                self.nan_value = value
                break

    def __missing__(self, key):
        if False:
            return 10
        if hasattr(self, 'nan_value') and is_scalar_nan(key):
            return self.nan_value
        raise KeyError(key)

def _map_to_integer(values, uniques):
    if False:
        return 10
    'Map values based on its position in uniques.'
    table = _nandict({val: i for (i, val) in enumerate(uniques)})
    return np.array([table[v] for v in values])

def _unique_python(values, *, return_inverse, return_counts):
    if False:
        return 10
    try:
        uniques_set = set(values)
        (uniques_set, missing_values) = _extract_missing(uniques_set)
        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted((t.__qualname__ for t in set((type(v) for v in values))))
        raise TypeError(f'Encoders require their input argument must be uniformly strings or numbers. Got {types}')
    ret = (uniques,)
    if return_inverse:
        ret += (_map_to_integer(values, uniques),)
    if return_counts:
        ret += (_get_counts(values, uniques),)
    return ret[0] if len(ret) == 1 else ret

def _encode(values, *, uniques, check_unknown=True):
    if False:
        i = 10
        return i + 15
    'Helper function to encode values into [0, n_uniques - 1].\n\n    Uses pure python method for object dtype, and numpy method for\n    all other dtypes.\n    The numpy method has the limitation that the `uniques` need to\n    be sorted. Importantly, this is not checked but assumed to already be\n    the case. The calling method needs to ensure this for all non-object\n    values.\n\n    Parameters\n    ----------\n    values : ndarray\n        Values to encode.\n    uniques : ndarray\n        The unique values in `values`. If the dtype is not object, then\n        `uniques` needs to be sorted.\n    check_unknown : bool, default=True\n        If True, check for values in `values` that are not in `unique`\n        and raise an error. This is ignored for object dtype, and treated as\n        True in this case. This parameter is useful for\n        _BaseEncoder._transform() to avoid calling _check_unknown()\n        twice.\n\n    Returns\n    -------\n    encoded : ndarray\n        Encoded values\n    '
    if values.dtype.kind in 'OUS':
        try:
            return _map_to_integer(values, uniques)
        except KeyError as e:
            raise ValueError(f'y contains previously unseen labels: {str(e)}')
    else:
        if check_unknown:
            diff = _check_unknown(values, uniques)
            if diff:
                raise ValueError(f'y contains previously unseen labels: {str(diff)}')
        return np.searchsorted(uniques, values)

def _check_unknown(values, known_values, return_mask=False):
    if False:
        return 10
    '\n    Helper function to check for unknowns in values to be encoded.\n\n    Uses pure python method for object dtype, and numpy method for\n    all other dtypes.\n\n    Parameters\n    ----------\n    values : array\n        Values to check for unknowns.\n    known_values : array\n        Known values. Must be unique.\n    return_mask : bool, default=False\n        If True, return a mask of the same shape as `values` indicating\n        the valid values.\n\n    Returns\n    -------\n    diff : list\n        The unique values present in `values` and not in `know_values`.\n    valid_mask : boolean array\n        Additionally returned if ``return_mask=True``.\n\n    '
    valid_mask = None
    if values.dtype.kind in 'OUS':
        values_set = set(values)
        (values_set, missing_in_values) = _extract_missing(values_set)
        uniques_set = set(known_values)
        (uniques_set, missing_in_uniques) = _extract_missing(uniques_set)
        diff = values_set - uniques_set
        nan_in_diff = missing_in_values.nan and (not missing_in_uniques.nan)
        none_in_diff = missing_in_values.none and (not missing_in_uniques.none)

        def is_valid(value):
            if False:
                for i in range(10):
                    print('nop')
            return value in uniques_set or (missing_in_uniques.none and value is None) or (missing_in_uniques.nan and is_scalar_nan(value))
        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = np.array([is_valid(value) for value in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
        diff = list(diff)
        if none_in_diff:
            diff.append(None)
        if nan_in_diff:
            diff.append(np.nan)
    else:
        unique_values = np.unique(values)
        diff = np.setdiff1d(unique_values, known_values, assume_unique=True)
        if return_mask:
            if diff.size:
                valid_mask = np.isin(values, known_values)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
        if np.isnan(known_values).any():
            diff_is_nan = np.isnan(diff)
            if diff_is_nan.any():
                if diff.size and return_mask:
                    is_nan = np.isnan(values)
                    valid_mask[is_nan] = 1
                diff = diff[~diff_is_nan]
        diff = list(diff)
    if return_mask:
        return (diff, valid_mask)
    return diff

class _NaNCounter(Counter):
    """Counter with support for nan values."""

    def __init__(self, items):
        if False:
            while True:
                i = 10
        super().__init__(self._generate_items(items))

    def _generate_items(self, items):
        if False:
            return 10
        'Generate items without nans. Stores the nan counts separately.'
        for item in items:
            if not is_scalar_nan(item):
                yield item
                continue
            if not hasattr(self, 'nan_count'):
                self.nan_count = 0
            self.nan_count += 1

    def __missing__(self, key):
        if False:
            return 10
        if hasattr(self, 'nan_count') and is_scalar_nan(key):
            return self.nan_count
        raise KeyError(key)

def _get_counts(values, uniques):
    if False:
        while True:
            i = 10
    'Get the count of each of the `uniques` in `values`.\n\n    The counts will use the order passed in by `uniques`. For non-object dtypes,\n    `uniques` is assumed to be sorted and `np.nan` is at the end.\n    '
    if values.dtype.kind in 'OU':
        counter = _NaNCounter(values)
        output = np.zeros(len(uniques), dtype=np.int64)
        for (i, item) in enumerate(uniques):
            with suppress(KeyError):
                output[i] = counter[item]
        return output
    (unique_values, counts) = _unique_np(values, return_counts=True)
    uniques_in_values = np.isin(uniques, unique_values, assume_unique=True)
    if np.isnan(unique_values[-1]) and np.isnan(uniques[-1]):
        uniques_in_values[-1] = True
    unique_valid_indices = np.searchsorted(unique_values, uniques[uniques_in_values])
    output = np.zeros_like(uniques, dtype=np.int64)
    output[uniques_in_values] = counts[unique_valid_indices]
    return output