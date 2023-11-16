import operator
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.ops.array_ops import comparison_op, na_logical_op

def test_na_logical_op_2d():
    if False:
        return 10
    left = np.arange(8).reshape(4, 2)
    right = left.astype(object)
    right[0, 0] = np.nan
    with pytest.raises(TypeError, match='unsupported operand type'):
        operator.or_(left, right)
    result = na_logical_op(left, right, operator.or_)
    expected = right
    tm.assert_numpy_array_equal(result, expected)

def test_object_comparison_2d():
    if False:
        for i in range(10):
            print('nop')
    left = np.arange(9).reshape(3, 3).astype(object)
    right = left.T
    result = comparison_op(left, right, operator.eq)
    expected = np.eye(3).astype(bool)
    tm.assert_numpy_array_equal(result, expected)
    right.flags.writeable = False
    result = comparison_op(left, right, operator.ne)
    tm.assert_numpy_array_equal(result, ~expected)