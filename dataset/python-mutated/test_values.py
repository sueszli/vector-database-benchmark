import numpy as np
import pytest
from pandas import IntervalIndex, Series, period_range
import pandas._testing as tm

class TestValues:

    @pytest.mark.parametrize('data', [period_range('2000', periods=4), IntervalIndex.from_breaks([1, 2, 3, 4])])
    def test_values_object_extension_dtypes(self, data):
        if False:
            return 10
        result = Series(data).values
        expected = np.array(data.astype(object))
        tm.assert_numpy_array_equal(result, expected)

    def test_values(self, datetime_series):
        if False:
            while True:
                i = 10
        tm.assert_almost_equal(datetime_series.values, list(datetime_series), check_dtype=False)