from typing import final
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_numeric_dtype

class BaseReduceTests:
    """
    Reduction specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if False:
            while True:
                i = 10
        res_op = getattr(ser, op_name)
        try:
            alt = ser.astype('float64')
        except (TypeError, ValueError):
            alt = ser.astype(object)
        exp_op = getattr(alt, op_name)
        if op_name == 'count':
            result = res_op()
            expected = exp_op()
        else:
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        if False:
            return 10
        return arr.dtype

    @final
    def check_reduce_frame(self, ser: pd.Series, op_name: str, skipna: bool):
        if False:
            print('Hello World!')
        arr = ser.array
        df = pd.DataFrame({'a': arr})
        kwargs = {'ddof': 1} if op_name in ['var', 'std'] else {}
        cmp_dtype = self._get_expected_reduction_dtype(arr, op_name, skipna)
        result1 = arr._reduce(op_name, skipna=skipna, keepdims=True, **kwargs)
        result2 = getattr(df, op_name)(skipna=skipna, **kwargs).array
        tm.assert_extension_array_equal(result1, result2)
        if not skipna and ser.isna().any():
            expected = pd.array([pd.NA], dtype=cmp_dtype)
        else:
            exp_value = getattr(ser.dropna(), op_name)()
            expected = pd.array([exp_value], dtype=cmp_dtype)
        tm.assert_extension_array_equal(result1, expected)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        if False:
            print('Hello World!')
        op_name = all_boolean_reductions
        ser = pd.Series(data)
        if not self._supports_reduction(ser, op_name):
            msg = '[Cc]annot perform|Categorical is not ordered for operation|does not support reduction|'
            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)
        else:
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        if False:
            print('Hello World!')
        op_name = all_numeric_reductions
        ser = pd.Series(data)
        if not self._supports_reduction(ser, op_name):
            msg = '[Cc]annot perform|Categorical is not ordered for operation|does not support reduction|'
            with pytest.raises(TypeError, match=msg):
                getattr(ser, op_name)(skipna=skipna)
        else:
            self.check_reduce(ser, op_name, skipna)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        if False:
            for i in range(10):
                print('nop')
        op_name = all_numeric_reductions
        ser = pd.Series(data)
        if not is_numeric_dtype(ser.dtype):
            pytest.skip('not numeric dtype')
        if op_name in ['count', 'kurt', 'sem']:
            pytest.skip(f'{op_name} not an array method')
        if not self._supports_reduction(ser, op_name):
            pytest.skip(f'Reduction {op_name} not supported for this dtype')
        self.check_reduce_frame(ser, op_name, skipna)

class BaseNoReduceTests(BaseReduceTests):
    """we don't define any reductions"""

class BaseNumericReduceTests(BaseReduceTests):

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        if op_name in ['any', 'all']:
            pytest.skip('These are tested in BaseBooleanReduceTests')
        return True

class BaseBooleanReduceTests(BaseReduceTests):

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if False:
            print('Hello World!')
        if op_name not in ['any', 'all']:
            pytest.skip('These are tested in BaseNumericReduceTests')
        return True