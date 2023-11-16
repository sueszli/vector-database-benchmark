import pytest
import numpy as np
import pandas as pd
import random
from bigdl.chronos.data import TSDataset
from bigdl.chronos.utils import LazyImport
RollDataset = LazyImport('bigdl.chronos.data.utils.roll_dataset.RollDataset')
from ... import op_torch

def get_ts_df():
    if False:
        print('Hello World!')
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value': np.random.randn(sample_num), 'id': np.array(['00'] * sample_num), 'extra feature': np.random.randn(sample_num)})
    return train_df

def get_multi_id_ts_df():
    if False:
        return 10
    sample_num = 100
    train_df = pd.DataFrame({'value': np.random.randn(sample_num), 'id': np.array(['00'] * 50 + ['01'] * 50), 'extra feature': np.random.randn(sample_num)})
    train_df['datetime'] = pd.date_range('1/1/2019', periods=sample_num)
    train_df.loc[50:100, 'datetime'] = pd.date_range('1/1/2019', periods=50)
    return train_df

@op_torch
class TestRollDataset:

    @staticmethod
    def assert_equal_with_tsdataset(df, horizon, lookback, feature_num=1):
        if False:
            for i in range(10):
                print('nop')
        extra_feature_col = None if feature_num == 0 else ['extra feature']
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', extra_feature_col=extra_feature_col, id_col='id', repair=False)
        tsdata.roll(lookback=lookback, horizon=horizon)
        if horizon == 0:
            x = tsdata.to_numpy()
        else:
            (x, y) = tsdata.to_numpy()
        roll_dataset = RollDataset(df=df, dt_col='datetime', freq=None, lookback=lookback, horizon=horizon, feature_col=tsdata.feature_col, target_col=tsdata.target_col, id_col=tsdata.id_col)
        assert len(roll_dataset) == len(x)
        for i in range(len(x)):
            if horizon != 0:
                (xi, yi) = (x[i], y[i])
                (roll_dataset_xi, roll_dataset_yi) = roll_dataset[i]
                np.testing.assert_array_almost_equal(xi, roll_dataset_xi.detach().numpy())
                np.testing.assert_array_almost_equal(yi, roll_dataset_yi.detach().numpy())
            else:
                xi = x[i]
                roll_dataset_xi = roll_dataset[i]
                np.testing.assert_array_almost_equal(xi, roll_dataset_xi.detach().numpy())

    @staticmethod
    def combination_tests_for_df(df):
        if False:
            print('Hello World!')
        lookback = random.randint(1, 20)
        horizon_tests = [random.randint(1, 10), [1, 4, 16], 0]
        feature_num_tests = [0, 1]
        for horizon in horizon_tests:
            for feature_num in feature_num_tests:
                TestRollDataset.assert_equal_with_tsdataset(df=df, horizon=horizon, lookback=lookback, feature_num=feature_num)

    def test_single_id(self):
        if False:
            print('Hello World!')
        df = get_ts_df()
        TestRollDataset.combination_tests_for_df(df)

    def test_multi_id(self):
        if False:
            while True:
                i = 10
        df = get_multi_id_ts_df()
        TestRollDataset.combination_tests_for_df(df)

    def test_df_nan(self):
        if False:
            i = 10
            return i + 15
        df = get_ts_df()
        df['value'][0] = np.nan
        with pytest.raises(RuntimeError):
            RollDataset(df=df, dt_col='datetime', freq=None, lookback=2, horizon=1, feature_col=['extra feature'], target_col=['value'], id_col='id')