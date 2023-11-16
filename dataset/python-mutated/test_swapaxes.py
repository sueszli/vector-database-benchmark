import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm

class TestSwapAxes:

    def test_swapaxes(self):
        if False:
            print('Hello World!')
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(df.T, df.swapaxes(0, 1))
            tm.assert_frame_equal(df.T, df.swapaxes(1, 0))

    def test_swapaxes_noop(self):
        if False:
            i = 10
            return i + 15
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(df, df.swapaxes(0, 0))

    def test_swapaxes_invalid_axis(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            msg = 'No axis named 2 for object type DataFrame'
            with pytest.raises(ValueError, match=msg):
                df.swapaxes(2, 5)

    def test_round_empty_not_input(self):
        if False:
            for i in range(10):
                print('nop')
        df = DataFrame({'a': [1, 2]})
        msg = "'DataFrame.swapaxes' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.swapaxes('index', 'index')
        tm.assert_frame_equal(df, result)
        assert df is not result