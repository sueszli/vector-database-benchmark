from typing import Any
import numpy as np

def _dict_like(x):
    if False:
        print('Hello World!')
    'Returns true if an object is a dict or convertible to one, false if not.'
    try:
        _ = dict(x)
    except (TypeError, ValueError):
        return False
    return True

def _enumerable(x):
    if False:
        print('Hello World!')
    'Returns true if an object is enumerable, false if not.'
    try:
        _ = enumerate(x)
    except (TypeError, ValueError):
        return False
    return True

def assert_all_finite(x: Any, keypath=''):
    if False:
        i = 10
        return i + 15
    'Ensures that all scalars at all levels of the dictionary, list, array, or scalar are finite.\n\n    keypath is only used for logging error messages, to indicate where the non-finite value was detected.\n    '
    path_description = f' at {keypath} ' if keypath else ' '
    if np.isscalar(x):
        assert np.isfinite(x), f'Value{path_description}should be finite, but is {str(x)}.'
    elif isinstance(x, np.ndarray):
        non_finite_indices = np.nonzero(~np.isfinite(x))
        non_finite_values = x[non_finite_indices]
        assert np.all(np.isfinite(x)), f'All values{path_description}should be finite, but found {str(non_finite_values)} at positions {{str(np.array(non_finite_indices).flatten())}}.'
    elif _dict_like(x):
        for (k, v) in dict(x).items():
            assert_all_finite(v, keypath=keypath + '.' + str(k) if keypath else str(k))
    elif _enumerable(x):
        for (i, v) in enumerate(x):
            assert_all_finite(v, keypath=keypath + f'[{i}]')
    else:
        assert False, f'Unhandled type {str(type(x))} for value{path_description}'