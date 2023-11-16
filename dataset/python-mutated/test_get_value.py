import pytest
from pandas import DataFrame, MultiIndex

class TestGetValue:

    def test_get_set_value_no_partial_indexing(self):
        if False:
            while True:
                i = 10
        index = MultiIndex.from_tuples([(0, 1), (0, 2), (1, 1), (1, 2)])
        df = DataFrame(index=index, columns=range(4))
        with pytest.raises(KeyError, match='^0$'):
            df._get_value(0, 1)

    def test_get_value(self, float_frame):
        if False:
            return 10
        for idx in float_frame.index:
            for col in float_frame.columns:
                result = float_frame._get_value(idx, col)
                expected = float_frame[col][idx]
                assert result == expected