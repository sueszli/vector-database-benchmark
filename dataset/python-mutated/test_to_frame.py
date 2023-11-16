from pandas import DataFrame, Index, date_range
import pandas._testing as tm

class TestToFrame:

    def test_to_frame_datetime_tz(self):
        if False:
            print('Hello World!')
        idx = date_range(start='2019-01-01', end='2019-01-30', freq='D', tz='UTC')
        result = idx.to_frame()
        expected = DataFrame(idx, index=idx)
        tm.assert_frame_equal(result, expected)

    def test_to_frame_respects_none_name(self):
        if False:
            print('Hello World!')
        idx = date_range(start='2019-01-01', end='2019-01-30', freq='D', tz='UTC')
        result = idx.to_frame(name=None)
        exp_idx = Index([None], dtype=object)
        tm.assert_index_equal(exp_idx, result.columns)
        result = idx.rename('foo').to_frame(name=None)
        exp_idx = Index([None], dtype=object)
        tm.assert_index_equal(exp_idx, result.columns)