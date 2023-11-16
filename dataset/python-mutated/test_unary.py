import pytest
from pandas import Series
import pandas._testing as tm

class TestSeriesUnaryOps:

    def test_neg(self):
        if False:
            for i in range(10):
                print('nop')
        ser = tm.makeStringSeries()
        ser.name = 'series'
        tm.assert_series_equal(-ser, -1 * ser)

    def test_invert(self):
        if False:
            return 10
        ser = tm.makeStringSeries()
        ser.name = 'series'
        tm.assert_series_equal(-(ser < 0), ~(ser < 0))

    @pytest.mark.parametrize('source, neg_target, abs_target', [([1, 2, 3], [-1, -2, -3], [1, 2, 3]), ([1, 2, None], [-1, -2, None], [1, 2, None])])
    def test_all_numeric_unary_operators(self, any_numeric_ea_dtype, source, neg_target, abs_target):
        if False:
            return 10
        dtype = any_numeric_ea_dtype
        ser = Series(source, dtype=dtype)
        (neg_result, pos_result, abs_result) = (-ser, +ser, abs(ser))
        if dtype.startswith('U'):
            neg_target = -Series(source, dtype=dtype)
        else:
            neg_target = Series(neg_target, dtype=dtype)
        abs_target = Series(abs_target, dtype=dtype)
        tm.assert_series_equal(neg_result, neg_target)
        tm.assert_series_equal(pos_result, ser)
        tm.assert_series_equal(abs_result, abs_target)

    @pytest.mark.parametrize('op', ['__neg__', '__abs__'])
    def test_unary_float_op_mask(self, float_ea_dtype, op):
        if False:
            i = 10
            return i + 15
        dtype = float_ea_dtype
        ser = Series([1.1, 2.2, 3.3], dtype=dtype)
        result = getattr(ser, op)()
        target = result.copy(deep=True)
        ser[0] = None
        tm.assert_series_equal(result, target)