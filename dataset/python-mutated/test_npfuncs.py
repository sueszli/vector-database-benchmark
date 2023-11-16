import numpy as np
from pandas import date_range
import pandas._testing as tm

class TestSplit:

    def test_split_non_utc(self):
        if False:
            i = 10
            return i + 15
        indices = date_range('2016-01-01 00:00:00+0200', freq='s', periods=10)
        result = np.split(indices, indices_or_sections=[])[0]
        expected = indices._with_freq(None)
        tm.assert_index_equal(result, expected)