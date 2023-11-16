"""
Templates for invalid operations.
"""
from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from pandas._typing import npt

def invalid_comparison(left, right, op) -> npt.NDArray[np.bool_]:
    if False:
        return 10
    '\n    If a comparison has mismatched types and is not necessarily meaningful,\n    follow python3 conventions by:\n\n        - returning all-False for equality\n        - returning all-True for inequality\n        - raising TypeError otherwise\n\n    Parameters\n    ----------\n    left : array-like\n    right : scalar, array-like\n    op : operator.{eq, ne, lt, le, gt}\n\n    Raises\n    ------\n    TypeError : on inequality comparisons\n    '
    if op is operator.eq:
        res_values = np.zeros(left.shape, dtype=bool)
    elif op is operator.ne:
        res_values = np.ones(left.shape, dtype=bool)
    else:
        typ = type(right).__name__
        raise TypeError(f'Invalid comparison between dtype={left.dtype} and {typ}')
    return res_values

def make_invalid_op(name: str):
    if False:
        return 10
    '\n    Return a binary method that always raises a TypeError.\n\n    Parameters\n    ----------\n    name : str\n\n    Returns\n    -------\n    invalid_op : function\n    '

    def invalid_op(self, other=None):
        if False:
            for i in range(10):
                print('nop')
        typ = type(self).__name__
        raise TypeError(f'cannot perform {name} with this index type: {typ}')
    invalid_op.__name__ = name
    return invalid_op