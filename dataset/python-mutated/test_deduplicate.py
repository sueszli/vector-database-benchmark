import pytest
import pandas as pd
import numpy as np
from unittest import TestCase
from bigdl.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe
from ... import op_torch, op_tf2

def get_duplicated_ugly_ts_df():
    if False:
        i = 10
        return i + 15
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan
    df['datetime'] = pd.date_range('1/1/2019', periods=50)
    for i in range(20):
        df.loc[len(df)] = df.loc[np.random.randint(0, 49)]
    return df

@op_torch
@op_tf2
class TestDeduplicateTimeSeries(TestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.df = get_duplicated_ugly_ts_df()

    def teardown_method(self, method):
        if False:
            i = 10
            return i + 15
        pass

    def test_deduplicate_timeseries_dataframe(self):
        if False:
            while True:
                i = 10
        with pytest.raises(RuntimeError):
            deduplicate_timeseries_dataframe(self.df, dt_col='z')
        with pytest.raises(RuntimeError):
            deduplicate_timeseries_dataframe(self.df, dt_col='a')
        res_df = deduplicate_timeseries_dataframe(self.df, dt_col='datetime')
        assert len(res_df) == 50