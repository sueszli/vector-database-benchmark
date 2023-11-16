import pytest
import pandas as pd
import numpy as np
from unittest import TestCase
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe, _last_impute_timeseries_dataframe, _const_impute_timeseries_dataframe, _linear_impute_timeseries_dataframe
from ... import op_torch, op_tf2

def get_ugly_ts_df():
    if False:
        i = 10
        return i + 15
    data = np.random.random_sample((50, 5))
    mask = np.random.random_sample((50, 5))
    mask[mask >= 0.4] = 2
    mask[mask < 0.4] = 1
    mask[mask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan
    df['datetime'] = pd.date_range('1/1/2019', periods=50)
    return df

@op_torch
@op_tf2
class TestImputeTimeSeries(TestCase):

    def setup_method(self, method):
        if False:
            return 10
        self.df = get_ugly_ts_df()

    def teardown_method(self, method):
        if False:
            while True:
                i = 10
        pass

    def test_impute_timeseries_dataframe(self):
        if False:
            return 10
        with pytest.raises(RuntimeError):
            impute_timeseries_dataframe(self.df, dt_col='z')
        with pytest.raises(RuntimeError):
            impute_timeseries_dataframe(self.df, dt_col='datetime', mode='dummy')
        with pytest.raises(RuntimeError):
            impute_timeseries_dataframe(self.df, dt_col='a')
        last_res_df = impute_timeseries_dataframe(self.df, dt_col='datetime', mode='last')
        assert self.df.isna().sum().sum() != 0
        assert last_res_df.isna().sum().sum() == 0
        const_res_df = impute_timeseries_dataframe(self.df, dt_col='datetime', mode='const')
        assert self.df.isna().sum().sum() != 0
        assert const_res_df.isna().sum().sum() == 0
        linear_res_df = impute_timeseries_dataframe(self.df, dt_col='datetime', mode='linear')
        assert self.df.isna().sum().sum() != 0
        assert linear_res_df.isna().sum().sum() == 0

    def test_last_impute_timeseries_dataframe(self):
        if False:
            print('Hello World!')
        data = {'data': [np.nan, np.nan, 1, np.nan, 2, 3]}
        df = pd.DataFrame(data)
        res_df = _last_impute_timeseries_dataframe(df)
        assert res_df['data'][0] == 0
        assert res_df['data'][1] == 0
        assert res_df['data'][3] == 1

    def test_const_impute_timeseries_dataframe(self):
        if False:
            print('Hello World!')
        data = {'data': [np.nan, 1, np.nan, 2, 3]}
        df = pd.DataFrame(data)
        res_df = _const_impute_timeseries_dataframe(df, 1)
        assert res_df['data'][0] == 1
        assert res_df['data'][2] == 1

    def test_linear_timeseries_dataframe(self):
        if False:
            while True:
                i = 10
        data = {'data': [np.nan, 1, np.nan, 2, 3], 'datetime': pd.date_range('1/1/2019', periods=5)}
        df = pd.DataFrame(data)
        res_df = _linear_impute_timeseries_dataframe(df, dt_col='datetime')
        assert res_df['data'][0] == 1
        assert res_df['data'][2] == 1.5