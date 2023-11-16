#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
import numpy as np
import pandas as pd
import random
import tempfile
import os
import shutil

from unittest import TestCase
from bigdl.chronos.data import TSDataset

from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_almost_equal
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
from .. import op_torch, op_tf2, op_diff_set_all

def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def get_multi_id_ts_df():
    sample_num = 100
    train_df = pd.DataFrame({"value": np.random.randn(sample_num),
                             "id": np.array(['00']*50 + ['01']*50),
                             "extra feature": np.random.randn(sample_num)})
    train_df["datetime"] = pd.date_range('1/1/2019', periods=sample_num)
    train_df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    return train_df


def get_multi_id_ts_df_interval():
    sample_num = 100
    train_df = pd.DataFrame({"value": np.random.randn(sample_num),
                             "id": np.array(['00']*50 + ['01']*50),
                             "extra feature": np.random.randn(sample_num)})
    train_df["datetime"] = pd.date_range('1/1/2019', periods=sample_num)
    train_df.loc[49, "datetime"] = '1/1/2020'
    train_df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    train_df.loc[99, "datetime"] = '1/1/2020'
    train_df["datetime"] = train_df["datetime"].astype('datetime64')
    return train_df


def get_ugly_ts_df():
    data = np.random.random_sample((100, 5))
    mask = np.random.random_sample((100, 5))
    newmask = mask.copy()
    mask[newmask >= 0.4] = 2
    mask[newmask < 0.4] = 1
    mask[newmask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'][0] = np.nan  # make sure column 'a' has a N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=100)
    df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
    df["id"] = np.array(['00']*50 + ['01']*50)
    return df


def get_int_target_df(freq="D"):
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num, freq=freq),
                             "value": np.array(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def get_non_dt():
    df = pd.DataFrame({"datetime": np.arange(100),
                       "id": np.array(['00']*100),
                       "value": np.random.randn(100),
                       "extra feature": np.random.randn(100)})
    return df


def get_not_aligned_df():
    df_val = pd.DataFrame({"id": np.array(['00']*20+['01']*30+['02']*50),
                           "value": np.random.randn(100),
                           "extra feature": np.random.randn(100)})
    data_sec = pd.DataFrame({"datetime": pd.date_range(
        start='1/1/2019 00:00:00', periods=20, freq='S')})
    data_min = pd.DataFrame({"datetime": pd.date_range(
        start='1/2/2019 00:00:00', periods=30, freq='H')})
    data_hou = pd.DataFrame({"datetime": pd.date_range(
        start='1/3/2019 00:00:00', periods=50, freq='D')})
    dt_val = pd.concat([data_sec, data_min, data_hou],
                       axis=0, ignore_index=True)
    df = pd.merge(left=dt_val, right=df_val, left_index=True, right_index=True)
    return df


def get_missing_df():
    data = np.random.random_sample((50, 5))
    mask = np.random.random_sample((50, 5))
    mask[mask >= 0.4] = 2
    mask[mask < 0.4] = 1
    mask[mask < 0.2] = 0
    data[mask == 0] = None
    data[mask == 1] = np.nan
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['a'] = np.nan  # make sure column 'a' has are all N/A
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    return df


def get_abnormal_df():
    data = np.random.random_sample((50, 5))
    # two abnormal data
    data[ data == data.max()] = 5
    data[ data == data.min()] = -5
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    return df


def get_multi_interval_df():
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df["datetime"] = pd.date_range('1/1/2019', periods=50)
    # modify datetime to generate a different interval
    df["datetime"][25:] = pd.date_range('1/1/2020', periods=25)
    return df


def get_non_std_dt_df():
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df["datetime"] = ["2022-1-1"]*50
    return df


class TestTSDataset(TestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @op_torch
    def test_tsdataset_initialization(self):
        df = get_ts_df()

        # legal input
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        assert tsdata._id_list == ['00']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                       extra_feature_col="extra feature", id_col="id")
        assert tsdata._id_list == ['00']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df.drop(columns=["id"]), dt_col="datetime",
                                       target_col=["value"], extra_feature_col="extra feature")
        assert tsdata._id_list == ['0']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        # illegal input
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature", id_col=0)
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col=0, target_col=["value"],
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=0,
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(0, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature", id_col="id")
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value1"],
                                           extra_feature_col="extra feature", id_col="id")

    @op_torch
    @op_diff_set_all
    def test_tsdataset_from_parquet(self):
        df = get_ts_df()

        configs = dict(dt_col="datetime",
                       target_col="value",
                       extra_feature_col=["extra feature"],
                       id_col="id")
        tsdata_pd = TSDataset.from_pandas(df, **configs)

        temp = tempfile.mkdtemp()
        try:
            path = os.path.join(temp, "test.parquet")
            df.to_parquet(path)

            tsdata_pq = TSDataset.from_parquet(path, **configs)

            pd.testing.assert_frame_equal(tsdata_pd.to_pandas(), tsdata_pq.to_pandas(),
                                          check_like=True)
        finally:
            shutil.rmtree(temp)

    @op_torch
    def test_tsdataset_initialization_multiple(self):
        df = get_multi_id_ts_df()
        # legal input
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        assert tsdata._id_list == ['00', '01']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                       extra_feature_col="extra feature",
                                       id_col="id", repair=False)
        assert tsdata._id_list == ['00', '01']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        tsdata = TSDataset.from_pandas(df.drop(columns=["id"]), dt_col="datetime",
                                       target_col=["value"],
                                       extra_feature_col="extra feature", repair=False)
        assert tsdata._id_list == ['0']
        assert tsdata.feature_col == ["extra feature"]
        assert tsdata.target_col == ["value"]
        assert tsdata.dt_col == "datetime"
        assert tsdata._is_pd_datetime

        # illegael input
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature",
                                           id_col=0, repair=False)
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col=0, target_col=["value"],
                                           extra_feature_col="extra feature",
                                           id_col="id", repair=False)
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=0,
                                           extra_feature_col="extra feature",
                                           id_col="id", repair=False)
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(0, dt_col="datetime", target_col=["value"],
                                           extra_feature_col="extra feature",
                                           id_col="id", repair=False)
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col=["value1"],
                                           extra_feature_col="extra feature",
                                           id_col="id", repair=False)

    @op_torch
    def test_tsdataset_roll_single_id(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        with pytest.raises(RuntimeError):
            tsdata.to_numpy()

        # roll train, diff input.
        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        # add extra_feature_col.
        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=["extra feature"], target_col="value")
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=["extra feature"], target_col="value", id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=[], target_col="value")
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 1)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon,
                    feature_col=[], target_col="value", id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 1)
        assert y.shape == (len(df)-lookback-horizon+1, horizon, 1)

        # roll test.
        horizon = 0
        lookback = random.randint(1, 20)

        tsdata.roll(lookback=lookback, horizon=horizon)
        x = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)

        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        x = tsdata.to_numpy()
        assert x.shape == (len(df)-lookback-horizon+1, lookback, 2)
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_roll_multi_id(self):
        df = get_multi_id_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)

        # test train
        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1), lookback, 4)
        assert y.shape == ((50-lookback-horizon+1), horizon, 2)

        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        # horizon list.
        horizon_list = [1, 3, 5]
        tsdata.roll(lookback=lookback, horizon=horizon_list)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-max(horizon_list)+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-max(horizon_list)+1)*2, len(horizon_list), 1)

        horizon_list = [1, 5, 9]
        tsdata.roll(lookback=lookback, horizon=horizon_list, id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-max(horizon_list)+1), lookback, 4)
        assert y.shape == ((50-lookback-max(horizon_list)+1), len(horizon_list), 2)

        # target multi.
        tsdata = TSDataset.from_pandas(df,
                                       dt_col="datetime",
                                       target_col=["value", "extra feature"],
                                       id_col="id",
                                       repair=False)
        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=False)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, 2)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 2)

        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_roll_order(self):
        df = pd.DataFrame({"datetime": np.array(['1/1/2019', '1/1/2019', '1/2/2019', '1/2/2019']),
                           "value": np.array([1.9, 2.3, 2.4, 2.6]),
                           "id": np.array(['00', '01', '00', '01']),
                           "extra feature1": np.array([1, 0, 3, 0]),
                           "extra feature2": np.array([2, 9, 4, 2])})
        tsdata = TSDataset.from_pandas(df,
                                       dt_col="datetime",
                                       target_col="value",
                                       extra_feature_col=["extra feature1", "extra feature2"],
                                       id_col="id")
        x, y = tsdata.roll(lookback=1, horizon=1, id_sensitive=False).to_numpy()
        assert x.shape == (2, 1, 3) and y.shape == (2, 1, 1)
        assert np.array_equal(x, np.array([[[1.9, 1, 2]], [[2.3, 0, 9]]], dtype=np.float32))
        assert np.array_equal(y, np.array([[[2.4]], [[2.6]]], dtype=np.float32))

        x, y = tsdata.roll(lookback=1, horizon=1, id_sensitive=True).to_numpy()
        assert x.shape == (1, 1, 6) and y.shape == (1, 1, 2)
        assert np.array_equal(x, np.array([[[1.9, 2.3, 1, 2, 0, 9]]], dtype=np.float32))
        assert np.array_equal(y, np.array([[[2.4, 2.6]]], dtype=np.float32))

    @op_torch
    def test_tsdata_roll_int_target(self):
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)
        df = get_int_target_df()
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value',
                                       extra_feature_col=['extra feature'], id_col="id")
        x, y = tsdata.roll(lookback=lookback, horizon=horizon).to_numpy()
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdata_roll_timeenc(self):
        horizon = random.randint(1, 9)
        lookback = random.randint(10, 20)

        # single-id test
        for freq in ["D", "2D"]:
            df = get_int_target_df(freq=freq)
            expected_sample_num = len(df) - horizon - lookback + 1
            tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
            x, y, x_time, y_time =\
                tsdata.roll(lookback=lookback, horizon=horizon,
                            time_enc=True, label_len=lookback-horizon).to_numpy()
            assert x.shape == (expected_sample_num, lookback, 1)
            assert y.shape == (expected_sample_num, lookback, 1)
            assert x_time.shape == (expected_sample_num, lookback, 3)
            assert y_time.shape == (expected_sample_num, lookback, 3)

        # multi-id test
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
        x, y, x_time, y_time =\
            tsdata.roll(lookback=lookback, horizon=horizon,
                        time_enc=True, label_len=lookback-horizon).to_numpy()
        assert x.shape[1:] == (lookback, 1)
        assert y.shape[1:] == (lookback, 1)
        assert x_time.shape[1:] == (lookback, 3)
        assert y_time.shape[1:] == (lookback, 3)
        assert x.shape[0] == y.shape[0] == x_time.shape[0] == y_time.shape[0]

    @op_torch
    def test_tsdata_roll_timeenc_predict(self):
        horizon = 10
        lookback = random.randint(10, 20)
        df = get_int_target_df()
        len_df = len(df)
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
        x, y, x_time, y_time =\
            tsdata.roll(lookback=lookback, horizon=horizon,
                        time_enc=True, label_len=5, is_predict=True).to_numpy()
        assert x.shape[1:] == (lookback, 1)
        assert y.shape[1:] == (5, 1)
        assert x_time.shape[1:] == (lookback, 3)
        assert y_time.shape[1:] == (15, 3)
        assert x.shape[0] == y.shape[0] == x_time.shape[0] == y_time.shape[0] == len(df) - lookback + 1

    @op_torch
    def test_tsdata_roll_timeenc_to_torch_data_loader(self):
        horizon = random.randint(1, 9)
        lookback = random.randint(10, 20)

        # single-id test
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
        dataloader =\
            tsdata.to_torch_data_loader(lookback=lookback, horizon=horizon,
                        time_enc=True, label_len=lookback-horizon)
        x, y, x_time, y_time = next(iter(dataloader))
        assert x.shape[1:] == (lookback, 1)
        assert y.shape[1:] == (lookback, 1)
        assert x_time.shape[1:] == (lookback, 3)
        assert y_time.shape[1:] == (lookback, 3)
        # white box check
        assert dataloader.dataset.data_stamp_arr.shape[0] == dataloader.dataset.arr.shape[0]

        # multiple-id test
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
        dataloader =\
            tsdata.to_torch_data_loader(lookback=lookback, horizon=horizon,
                        time_enc=True, label_len=lookback-horizon)
        x, y, x_time, y_time = next(iter(dataloader))
        assert x.shape[1:] == (lookback, 1)
        assert y.shape[1:] == (lookback, 1)
        assert x_time.shape[1:] == (lookback, 3)
        assert y_time.shape[1:] == (lookback, 3)
        # white box check
        assert dataloader.dataset.data_stamp_arr.shape[0] == dataloader.dataset.arr.shape[0]

    @op_torch
    def test_tsdata_roll_timeenc_to_torch_data_loader_predict(self):
        horizon = 10
        lookback = random.randint(10, 20)
        df = get_int_target_df()
        tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value', id_col="id")
        dataloader =\
            tsdata.to_torch_data_loader(lookback=lookback, horizon=horizon,
                        time_enc=True, label_len=5, is_predict=True, batch_size=len(df))
        x, y, x_time, y_time = next(iter(dataloader))
        assert x.shape[1:] == (lookback, 1)
        assert y.shape[1:] == (5, 1)
        assert x_time.shape[1:] == (lookback, 3)
        assert y_time.shape[1:] == (15, 3)
        assert x.shape[0] == y.shape[0] == x_time.shape[0] == y_time.shape[0] == len(df) - lookback + 1

    @op_torch
    def test_tsdataset_to_torch_loader_roll(self):
        df_single_id = get_ts_df()
        df_multi_id = get_multi_id_ts_df()
        for df in [df_single_id, df_multi_id]:
            horizon = random.randint(1, 10)
            lookback = random.randint(1, 20)
            batch_size = random.randint(16, 32)

            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                           extra_feature_col=["extra feature"], id_col="id")

            # train
            torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=horizon)
            for x_batch, y_batch in torch_loader:
                assert tuple(x_batch.size()) == (batch_size, lookback, 2)
                assert tuple(y_batch.size()) == (batch_size, horizon, 1)
                break

            # test
            torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=0)
            for x_batch in torch_loader:
                assert tuple(x_batch.size()) == (batch_size, lookback, 2)
                break

            # specify feature_col
            torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=horizon,
                                                       feature_col=[])
            for x_batch, y_batch in torch_loader:
                assert tuple(x_batch.size()) == (batch_size, lookback, 1)
                assert tuple(y_batch.size()) == (batch_size, horizon, 1)
                break

            # Non-subset relationship
            with pytest.raises(ValueError):
                tsdata.to_torch_data_loader(batch_size=batch_size,
                                            lookback=lookback,
                                            horizon=horizon,
                                            target_col=['value', 'extra feature'])

            # specify horizon_list
            horizon_list = [1, 3, 5]
            torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=horizon_list)
            for x_batch, y_batch in torch_loader:
                assert tuple(x_batch.size()) == (batch_size, lookback, 2)
                assert tuple(y_batch.size()) == (batch_size, len(horizon_list), 1)
                break

            # multi target_col
            tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                           target_col=["value", "extra feature"], id_col="id")
            torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=horizon)
            for x_batch, y_batch in torch_loader:
                assert tuple(x_batch.size()) == (batch_size, lookback, 2)
                assert tuple(y_batch.size()) == (batch_size, horizon, 2)
                break

    @op_torch
    def test_tsdataset_to_torch_loader(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)
        batch_size = random.randint(16, 32)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        with pytest.raises(RuntimeError):
            tsdata.to_torch_data_loader(roll=False)

        tsdata.roll(lookback=lookback, horizon=horizon)
        loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                             roll=False,
                                             lookback=lookback,
                                             horizon=horizon)
        for x_batch, y_batch in loader:
            assert tuple(x_batch.size()) == (batch_size, lookback, 2)
            assert tuple(y_batch.size()) == (batch_size, horizon, 1)
            break

    @op_torch
    def test_tsdataset_to_torch_loader_lessthansample(self):
        lookback = 96
        horizon = 48
        df = pd.DataFrame(np.random.randint(1, 10, size=(100, 1)), columns=["target"])
        df.insert(0, "datetime", pd.date_range(start="2022-7-22", periods=100, freq="H"))
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="target")
        with pytest.raises(RuntimeError):
            tsdata.to_torch_data_loader(lookback=lookback, horizon=horizon)

    @op_torch
    def test_tsdata_multi_unscale_numpy_torch_load(self):
        lookback = random.randint(1, 10)
        horizon = random.randint(1, 20)
        batch_size = random.randint(16, 32)
        df = get_multi_id_ts_df()
        df_test = get_multi_id_ts_df()
        tsdata_train = TSDataset.from_pandas(df,
                                             target_col='value',
                                             dt_col='datetime',
                                             extra_feature_col='extra feature',
                                             id_col='id')
        tsdata_test = TSDataset.from_pandas(df_test,
                                            target_col='value',
                                            dt_col='datetime',
                                            extra_feature_col='extra feature',
                                            id_col='id')
        # roll is True.
        from sklearn.preprocessing import StandardScaler
        stand = StandardScaler()
        for tsdata in [tsdata_train, tsdata_test]:
            tsdata.scale(stand, fit=tsdata is tsdata_train)

        test_loader = tsdata_test.to_torch_data_loader(batch_size=batch_size,
                                                       lookback=lookback,
                                                       horizon=horizon)
        import torch
        from torch.utils.data.dataloader import DataLoader
        test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)

        batch_load_list = []
        for _, y_batch in test_loader:
            batch_load_list.append(y_batch)
        y_test = torch.cat(batch_load_list, dim=0)
        pred = np.copy(y_test.numpy())  # sanity check

        unscaled_pred = tsdata_train.unscale_numpy(pred)
        unscaled_y_test = tsdata_train.unscale_numpy(y_test.numpy())

        _, unscaled_y_test_reproduce = tsdata_test.unscale()\
                                                  .roll(lookback=lookback, horizon=horizon)\
                                                  .to_numpy()

        assert_array_almost_equal(unscaled_pred, unscaled_y_test_reproduce)
        assert_array_almost_equal(unscaled_y_test, unscaled_y_test_reproduce)

        tsdata._check_basic_invariants()

    @op_tf2
    def test_tsdata_to_tf_dataset(self):
        df = get_ts_df()
        batch_size, lookback, horizon = 32, 10, 1
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        data = tsdata.roll(lookback=lookback, horizon=horizon).to_tf_dataset(batch_size=batch_size)
        val = next(iter(data))
        assert val[0].numpy().shape == (batch_size, lookback, 2)
        assert val[1].numpy().shape == (batch_size, horizon, 1)

    @op_torch
    def test_tsdataset_imputation(self):
        for val in ["last", "const", "linear"]:
            df = get_ugly_ts_df()
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="e",
                                           extra_feature_col=["a", "b", "c", "d"],
                                           id_col="id", repair=False)
            tsdata.impute(mode=val)
            assert tsdata.to_pandas().isna().sum().sum() == 0
            assert len(tsdata.to_pandas()) == 100
            tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_deduplicate(self):
        df = get_ugly_ts_df()
        for _ in range(20):
            df.loc[len(df)] = df.loc[np.random.randint(0, 99)]
        assert len(df) == 120
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="e",
                                       extra_feature_col=["a", "b", "c", "d"],
                                       id_col="id", repair=False)
        tsdata.deduplicate()
        assert len(tsdata.to_pandas()) == 100
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_datetime_feature(self):
        df = get_ts_df()
        # interval = day
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_dt_feature()
        assert set(tsdata.to_pandas().columns) == {'DAY',
                                                   'IS_WEEKEND',
                                                   'WEEKDAY',
                                                   'MONTH',
                                                   'YEAR',
                                                   'DAYOFYEAR',
                                                   'WEEKOFYEAR',
                                                   'extra feature',
                                                   'value',
                                                   'datetime',
                                                   'id'}
        assert set(tsdata.feature_col) == {'DAY',
                                           'IS_WEEKEND',
                                           'WEEKDAY',
                                           'MONTH',
                                           'YEAR',
                                           'DAYOFYEAR',
                                           'WEEKOFYEAR',
                                           'extra feature'}
        tsdata._check_basic_invariants()

        # interval = day, one_hot = ["WEEKDAY"]
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_dt_feature(one_hot_features=["WEEKDAY"])
        assert set(tsdata.to_pandas().columns) == {'DAY',
                                                   'IS_WEEKEND',
                                                   'WEEKDAY_0',
                                                   'WEEKDAY_1',
                                                   'WEEKDAY_2',
                                                   'WEEKDAY_3',
                                                   'WEEKDAY_4',
                                                   'WEEKDAY_5',
                                                   'WEEKDAY_6',
                                                   'MONTH',
                                                   'YEAR',
                                                   'DAYOFYEAR',
                                                   'WEEKOFYEAR',
                                                   'extra feature',
                                                   'value',
                                                   'datetime',
                                                   'id'}
        assert set(tsdata.feature_col) == {'DAY',
                                           'IS_WEEKEND',
                                           'WEEKDAY_0',
                                           'WEEKDAY_1',
                                           'WEEKDAY_2',
                                           'WEEKDAY_3',
                                           'WEEKDAY_4',
                                           'WEEKDAY_5',
                                           'WEEKDAY_6',
                                           'MONTH',
                                           'YEAR',
                                           'DAYOFYEAR',
                                           'WEEKOFYEAR',
                                           'extra feature'}
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_datetime_feature_multiple(self):
        df = get_multi_id_ts_df()
        # interval = day
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_dt_feature()
        assert set(tsdata.to_pandas().columns) == {'DAY',
                                                   'IS_WEEKEND',
                                                   'WEEKDAY',
                                                   'MONTH',
                                                   'YEAR',
                                                   'DAYOFYEAR',
                                                   'WEEKOFYEAR',
                                                   'extra feature',
                                                   'value',
                                                   'datetime',
                                                   'id'}
        assert set(tsdata.feature_col) == {'DAY',
                                           'IS_WEEKEND',
                                           'WEEKDAY',
                                           'MONTH',
                                           'YEAR',
                                           'DAYOFYEAR',
                                           'WEEKOFYEAR',
                                           'extra feature'}
        tsdata._check_basic_invariants()

        # interval = day, one_hot = ["WEEKDAY"]
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_dt_feature(one_hot_features=["WEEKDAY"])
        assert set(tsdata.to_pandas().columns) == {'DAY',
                                                   'IS_WEEKEND',
                                                   'WEEKDAY_0',
                                                   'WEEKDAY_1',
                                                   'WEEKDAY_2',
                                                   'WEEKDAY_3',
                                                   'WEEKDAY_4',
                                                   'WEEKDAY_5',
                                                   'WEEKDAY_6',
                                                   'MONTH',
                                                   'YEAR',
                                                   'DAYOFYEAR',
                                                   'WEEKOFYEAR',
                                                   'extra feature',
                                                   'value',
                                                   'datetime',
                                                   'id'}
        assert set(tsdata.feature_col) == {'DAY',
                                           'IS_WEEKEND',
                                           'WEEKDAY_0',
                                           'WEEKDAY_1',
                                           'WEEKDAY_2',
                                           'WEEKDAY_3',
                                           'WEEKDAY_4',
                                           'WEEKDAY_5',
                                           'WEEKDAY_6',
                                           'MONTH',
                                           'YEAR',
                                           'DAYOFYEAR',
                                           'WEEKOFYEAR',
                                           'extra feature'}
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_scale_unscale(self):
        df = get_ts_df()
        df_test = get_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata_test = TSDataset.from_pandas(df_test, dt_col="datetime", target_col="value",
                                            extra_feature_col=["extra feature"], id_col="id")

        from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        scalers = [StandardScaler(), MaxAbsScaler(),
                   MinMaxScaler(), RobustScaler()]
        for scaler in scalers:
            tsdata.scale(scaler)
            tsdata_test.scale(scaler, fit=False)

            with pytest.raises(AssertionError):
                assert_frame_equal(tsdata.to_pandas(), df)
            with pytest.raises(AssertionError):
                assert_frame_equal(tsdata_test.to_pandas(), df_test)

            tsdata.unscale()
            tsdata_test.unscale()

            assert_frame_equal(tsdata.to_pandas(), df)
            assert_frame_equal(tsdata_test.to_pandas(), df_test)

        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_unscale_numpy(self):
        df = get_multi_id_ts_df()
        df_test = get_multi_id_ts_df()

        from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        scalers = [StandardScaler(),
                   StandardScaler(with_mean=False),
                   StandardScaler(with_std=False),
                   MaxAbsScaler(),
                   MinMaxScaler(),
                   MinMaxScaler(feature_range=(1, 3)),
                   RobustScaler(),
                   RobustScaler(with_centering=False),
                   RobustScaler(with_scaling=False),
                   RobustScaler(quantile_range=(20, 80))]

        for scaler in scalers:
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                           extra_feature_col=["extra feature"], id_col="id")
            tsdata_test = TSDataset.from_pandas(df_test, dt_col="datetime", target_col="value",
                                                extra_feature_col=["extra feature"], id_col="id")
            tsdata.gen_dt_feature()\
                  .scale(scaler)\
                  .roll(lookback=5, horizon=4, id_sensitive=True)
            tsdata_test.gen_dt_feature()\
                       .scale(scaler, fit=False)\
                       .roll(lookback=5, horizon=4, id_sensitive=True)

            _, _ = tsdata.to_numpy()
            _, y_test = tsdata_test.to_numpy()

            pred = np.copy(y_test)  # sanity check

            unscaled_pred = tsdata.unscale_numpy(pred)
            unscaled_y_test = tsdata.unscale_numpy(y_test)
            tsdata_test.unscale()\
                       .roll(lookback=5, horizon=4, id_sensitive=True)
            _, unscaled_y_test_reproduce = tsdata_test.to_numpy()

            assert_array_almost_equal(unscaled_pred, unscaled_y_test_reproduce)
            assert_array_almost_equal(unscaled_y_test, unscaled_y_test_reproduce)

            tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_resample(self):
        df = get_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.resample('2D', df["datetime"][0], df["datetime"][df.shape[0]-1])
        assert len(tsdata.to_pandas()) == (df.shape[0] + 1) // 2
        tsdata._check_basic_invariants()

        # target_col\extra_feature_col dtype is object(str).
        sample_num = np.random.randint(100, 200)
        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                           "value": np.array(['test_value']*sample_num),
                           "id": np.array(['00']*sample_num),
                           "extra feature": np.array(['test_extra_feature']*sample_num)})
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        with pytest.raises(RuntimeError):
            tsdata.resample('2S', df.datetime[0], df.datetime[df.shape[0]-1])
        tsdata._check_basic_invariants()

        # target_col\extra_feature_col dtype is object(numeric).
        df = get_ts_df()
        df.value = df.value.astype(object)
        df['extra feature'] = df['extra feature'].astype(object)
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        before_sampling = tsdata.df.columns
        tsdata.resample('2S', df.datetime[0], df.datetime[df.shape[0]-1])
        assert set(before_sampling) == set(tsdata.df.columns)
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_resample_multiple(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        tsdata.resample('2D', df["datetime"][0], df["datetime"][df.shape[0]-1])
        assert len(tsdata.to_pandas()) == df.shape[0] // 2
        tsdata._check_basic_invariants()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        tsdata.resample('2D')
        assert len(tsdata.to_pandas()) == 50
        tsdata._check_basic_invariants()

        # target_col\extra_feature_col dtype is object(str).
        df = pd.DataFrame({"value": np.array(['test_value']*100),
                           "id": np.array(['00']*50 + ['01']*50),
                           "extra feature": np.array(['test_extra_feature']*100)})
        df["datetime"] = pd.date_range('1/1/2019', periods=100)
        df.loc[50:100, "datetime"] = pd.date_range('1/1/2019', periods=50)
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        with pytest.raises(RuntimeError):
            tsdata.resample('2S', df.datetime[0], df.datetime[df.shape[0]-1])
        tsdata._check_basic_invariants()

        # target_col/extra_feature_col dtype is object(numeric).
        df = get_multi_id_ts_df()
        df.value = df.value.astype(object)
        df['extra feature'] = df['extra feature'].astype(object)
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        before_sampling = tsdata.df.columns
        tsdata.resample('2S', df.datetime[0], df.datetime[df.shape[0]-1])
        assert set(before_sampling) == set(tsdata.df.columns)
        tsdata._check_basic_invariants()

    @op_torch
    def test_tsdataset_split(self):
        df = get_ts_df()
        # only train and test
        tsdata_train, tsdata_valid, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                  extra_feature_col=["extra feature"], id_col="id",
                                  with_split=True, val_ratio=0, test_ratio=0.1)
        # standard split with all three sets
        tsdata_train, tsdata_valid, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                  extra_feature_col=["extra feature"], id_col="id",
                                  with_split=True, val_ratio=0.1, test_ratio=0.1)

        assert set(np.unique(tsdata_train.to_pandas()["id"])) == {"00"}
        assert set(np.unique(tsdata_valid.to_pandas()["id"])) == {"00"}
        assert set(np.unique(tsdata_test.to_pandas()["id"])) == {"00"}

        assert len(tsdata_train.to_pandas()) == df[:-(int(df.shape[0]*0.1)*2)].shape[0]
        assert len(tsdata_valid.to_pandas()) == int(df.shape[0] * 0.1)
        assert len(tsdata_test.to_pandas()) == int(df.shape[0] * 0.1)
        tsdata_train.feature_col.append("new extra feature")
        assert len(tsdata_train.feature_col) == 2
        assert len(tsdata_valid.feature_col) == 1
        assert len(tsdata_test.feature_col) == 1

        tsdata_train.target_col[0] = "new value"
        assert tsdata_train.target_col[0] == "new value"
        assert tsdata_valid.target_col[0] != "new value"
        assert tsdata_test.target_col[0] != "new value"

    @op_torch
    def test_tsdataset_split_multiple(self):
        df = get_multi_id_ts_df()
        tsdata_train, tsdata_valid, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                  extra_feature_col=["extra feature"], id_col="id",
                                  with_split=True, val_ratio=0.1, test_ratio=0.1,
                                  repair=False)

        assert set(np.unique(tsdata_train.to_pandas()["id"])) == {"00", "01"}
        assert set(np.unique(tsdata_valid.to_pandas()["id"])) == {"00", "01"}
        assert set(np.unique(tsdata_test.to_pandas()["id"])) == {"00", "01"}

        assert len(tsdata_train.to_pandas()) == (50 * 0.8)*2
        assert len(tsdata_valid.to_pandas()) == (50 * 0.1)*2
        assert len(tsdata_test.to_pandas()) == (50 * 0.1)*2

        assert tsdata_train.feature_col is not tsdata_valid.feature_col
        assert tsdata_train.feature_col is not tsdata_test.feature_col
        assert tsdata_train.target_col is not tsdata_valid.target_col
        assert tsdata_train.target_col is not tsdata_test.target_col

        tsdata_train.feature_col.append("new extra feature")
        assert len(tsdata_train.feature_col) == 2
        assert len(tsdata_valid.feature_col) == 1
        assert len(tsdata_test.feature_col) == 1

        tsdata_train.target_col[0] = "new value"
        assert tsdata_train.target_col[0] == "new value"
        assert tsdata_valid.target_col[0] != "new value"
        assert tsdata_test.target_col[0] != "new value"

    @op_torch
    @op_diff_set_all
    def test_tsdataset_global_feature(self):
        for val in ["minimal"]:
            df = get_ts_df()
            tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                           extra_feature_col=["extra feature"], id_col="id")
            tsdata.gen_global_feature(settings=val)
            tsdata._check_basic_invariants()

    @op_torch
    @op_diff_set_all
    def test_tsdataset_global_feature_multiple(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_global_feature(settings="minimal")
        tsdata._check_basic_invariants()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")
        tsdata.gen_global_feature(settings="minimal", n_jobs=2)
        tsdata._check_basic_invariants()

    @op_torch
    @op_diff_set_all
    def test_tsdataset_rolling_feature_multiple(self):
        df = get_multi_id_ts_df()
        horizon = random.randint(2, 10)
        lookback = random.randint(2, 20)
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        tsdata.gen_rolling_feature(settings="minimal", window_size=lookback)
        tsdata._check_basic_invariants()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"],
                                       id_col="id", repair=False)
        tsdata.gen_rolling_feature(settings="minimal", window_size=lookback, n_jobs=2)
        tsdata._check_basic_invariants()

        # roll train
        tsdata.roll(lookback=lookback, horizon=horizon)
        x, y = tsdata.to_numpy()
        feature_num = len(tsdata.feature_col) + len(tsdata.target_col)
        assert x.shape == ((50-lookback-horizon+1)*2, lookback, feature_num)
        assert y.shape == ((50-lookback-horizon+1)*2, horizon, 1)

        tsdata.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        x, y = tsdata.to_numpy()
        assert x.shape == ((50-lookback-horizon+1), lookback, feature_num*2)
        assert y.shape == ((50-lookback-horizon+1), horizon, 2)

        tsdata._check_basic_invariants()

    @op_torch
    def test_check_scale_sequence(self):
        df = get_multi_id_ts_df()
        # with split is True.
        td_train, td_valid, td_test = TSDataset.from_pandas(df, dt_col="datetime",
                                                            target_col="value",
                                                            extra_feature_col=[
                                                                "extra feature"],
                                                            id_col="id",
                                                            with_split=True,
                                                            val_ratio=0.1,
                                                            test_ratio=0.1)
        from sklearn.preprocessing import StandardScaler
        stand = StandardScaler()
        with pytest.raises(RuntimeError):
            for tsdata in [td_train, td_valid, td_test]:
                tsdata.scale(stand, fit=False)
            tsdata._check_basic_invariants()

        # remove due to the possible large cost on test sys
        # with pytest.raises(RuntimeError):
        #     tsdata.gen_global_feature(settings="minimal")\
        #           .gen_rolling_feature(settings="minimal", window_size=5)

    @op_torch
    def test_non_pd_datetime(self):
        df = get_non_dt()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                       target_col="value",
                                       extra_feature_col="extra feature",
                                       id_col="id")

        with pytest.raises(RuntimeError):
            tsdata.resample('2D')
        with pytest.raises(RuntimeError):
            tsdata.gen_dt_feature()
        with pytest.raises(RuntimeError):
            tsdata.gen_rolling_feature(settings="minimal", window_size=1000)

        tsdata._check_basic_invariants()

    @op_torch
    def test_not_aligned(self):
        df = get_not_aligned_df()
        tsdata = TSDataset.from_pandas(df, target_col="value",
                                       dt_col="datetime",
                                       extra_feature_col="extra feature",
                                       id_col="id",
                                       repair=False)
        with pytest.raises(RuntimeError):
            tsdata.roll(lookback=5, horizon=2, id_sensitive=True)
        tsdata._check_basic_invariants()

    @op_torch
    def test_dt_sorted(self):
        df = pd.DataFrame({"datetime": np.array(['20000101', '20000102', '20000102', '20000101']),
                           "value": np.array([1.9, 2.3, 2.4, 2.6]),
                           "id": np.array(['00', '01', '00', '01'])})

        tsdata = TSDataset.from_pandas(df, target_col='value',
                                       dt_col='datetime')
        with pytest.raises(RuntimeError):
            tsdata._check_basic_invariants(strict_check=True)

    @op_torch
    @op_diff_set_all
    def test_cycle_length_est(self):
        df = get_multi_id_ts_df()
        tsdata = TSDataset.from_pandas(df,
                                       target_col='value',
                                       dt_col='datetime',
                                       extra_feature_col='extra feature',
                                       id_col='id')

        with pytest.raises(RuntimeError):
            tsdata.get_cycle_length(aggregate="normal")
        with pytest.raises(RuntimeError):
            tsdata.get_cycle_length(aggregate=10)
        with pytest.raises(RuntimeError):
            tsdata.get_cycle_length(top_k='3')
        with pytest.raises(RuntimeError):
            tsdata.get_cycle_length(top_k=24)
        
        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=100),
                           "value": np.sin(np.array((0, 30, 45, 60, 90)*20)*np.pi/180),
                           "id": np.array(['00']*100),
                           "extra feature": np.random.randn(100)})
        tsdata = TSDataset.from_pandas(df,
                                       target_col='value',
                                       dt_col='datetime',
                                       extra_feature_col='extra feature')
        tsdata.roll(lookback='auto', horizon=1)
        df_x, _ = tsdata.to_numpy()
        assert df_x.shape[1] == 5

        tsdata.roll(lookback=tsdata.get_cycle_length(aggregate='median', top_k=4),
                    horizon=1)
        assert tsdata.best_cycle_length == 5

        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=100),
                           "value": np.ones(100)})
        tsdata = TSDataset.from_pandas(df,
                                       target_col='value',
                                       dt_col='datetime')
        # not meaningful, but no error should be raised.
        tsdata.get_cycle_length(aggregate='min', top_k=3)

    @op_torch
    def test_lookback_equal_to_one(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = 1

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for BaseForecaster
        data = tsdata.roll(lookback=lookback, horizon=horizon, is_predict=True).to_numpy()
        assert not isinstance(data, (list, tuple))

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for AutoformerForecaster
        data = tsdata.roll(lookback=lookback, horizon=horizon, time_enc=True, is_predict=True).to_numpy()
        assert len(data) == 4
        assert data[1].shape[1] == max(lookback // 2, 1)
        assert data[3].shape[1] == max(lookback // 2, 1) + horizon

    @op_torch
    def test_is_predict_for_roll_and_numpy(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for BaseForecaster
        data = tsdata.roll(lookback=lookback, horizon=horizon, is_predict=True).to_numpy()
        assert not isinstance(data, (list, tuple))

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for AutoformerForecaster
        data = tsdata.roll(lookback=lookback, horizon=horizon, time_enc=True, is_predict=True).to_numpy()
        assert len(data) == 4
        assert data[1].shape[1] == max(lookback // 2, 1)
        assert data[3].shape[1] == max(lookback // 2, 1) + horizon

    @op_torch
    def test_is_predict_for_roll_and_to_loader(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for BaseForecaster
        tsdata.roll(lookback=lookback, horizon=horizon, is_predict=True)
        loader = tsdata.to_torch_data_loader(batch_size=32,
                                             roll=False,)
        data = next(iter(loader))
        assert len(data) == 1

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for AutoformerForecaster
        tsdata.roll(lookback=lookback, horizon=horizon, time_enc=True, is_predict=True)
        loader = tsdata.to_torch_data_loader(batch_size=32,
                                             roll=False)
        data = next(iter(loader))
        assert len(data) == 4
        assert data[1].shape[1] == max(lookback // 2, 1)
        assert data[3].shape[1] == max(lookback // 2, 1) + horizon


    @op_torch
    def test_is_predict_for_to_loader(self):
        df = get_ts_df()
        horizon = random.randint(1, 10)
        lookback = random.randint(1, 20)

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for BaseForecaster
        loader = tsdata.to_torch_data_loader(lookback=lookback,
                                             horizon=horizon,
                                             roll=True,
                                             is_predict=True)
        data = next(iter(loader))
        assert not isinstance(data, (list, tuple))

        tsdata = TSDataset.from_pandas(df, dt_col="datetime", target_col="value",
                                       extra_feature_col=["extra feature"], id_col="id")

        # for AutoformerForecaster
        loader = tsdata.to_torch_data_loader(lookback=lookback,
                                             horizon=horizon,
                                             roll=True,
                                             time_enc=True,
                                             is_predict=True)
        data = next(iter(loader))
        assert len(data) == 4
        assert data[1].shape[1] == max(lookback // 2, 1)
        assert data[3].shape[1] == max(lookback // 2, 1) + horizon

    @op_torch
    def test_tsdataset_missing_check_and_repair(self):
        df = get_missing_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                       target_col=['a', 'b', 'c', 'd', 'e'],
                                       extra_feature_col=None,
                                       repair=True)
        missing_value = tsdata.df.isna().sum().sum()
        assert missing_value == 0

    @op_torch
    def test_tsdataset_non_std_dt_check_and_repair(self):
        df = get_non_std_dt_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                       target_col=['a', 'b', 'c', 'd', 'e'],
                                       extra_feature_col=None,
                                       repair=True)
        _is_pd_datetime = pd.api.types.is_datetime64_any_dtype(tsdata.df["datetime"].dtypes)
        assert _is_pd_datetime is True

    @op_torch
    def test_tsdataset_interval_check_and_repair(self):
        df = get_multi_interval_df()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                       target_col=['a', 'b', 'c', 'd', 'e'],
                                       extra_feature_col=None,
                                       repair=True)
        dt_col = tsdata.df["datetime"]
        interval = dt_col.shift(-1) - dt_col
        unique_intervals = interval[:-1].unique()
        assert len(unique_intervals) == 1

    @op_torch
    def test_tsdataset_interval_repair_for_single_and_multi_id(self):
        df = get_multi_id_ts_df_interval()
        tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                       target_col=['value'],
                                       extra_feature_col=None,
                                       repair=True)
        dt_col = tsdata.df["datetime"]
        assert len(dt_col) == 366
        df = get_multi_id_ts_df_interval()
        tsdata = TSDataset.from_pandas(df,
                                       id_col="id",
                                       dt_col="datetime",
                                       target_col=['value'],
                                       extra_feature_col=None,
                                       repair=True)
        dt_col = tsdata.df["datetime"]
        assert len(dt_col) == 366*2

    @op_torch
    def test_tsdataset_interval_check_and_repair_for_multi_id(self):
        df_multi_id = get_multi_id_ts_df()
        horizon = 10
        lookback = 20
        batch_size = 32

        tsdata = TSDataset.from_pandas(df_multi_id, dt_col="datetime", target_col="value",
                                        extra_feature_col=["extra feature"], id_col="id",
                                        repair=True)

        # train
        torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                    lookback=lookback,
                                                    horizon=horizon)
        for x_batch, y_batch in torch_loader:
            assert tuple(x_batch.size()) == (batch_size, lookback, 2)
            assert tuple(y_batch.size()) == (batch_size, horizon, 1)
            break

        # test
        torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                    lookback=lookback,
                                                    horizon=0)
        for x_batch in torch_loader:
            assert tuple(x_batch.size()) == (batch_size, lookback, 2)
            break

        # specify feature_col
        torch_loader = tsdata.to_torch_data_loader(batch_size=batch_size,
                                                    lookback=lookback,
                                                    horizon=horizon,
                                                    feature_col=[])
        for x_batch, y_batch in torch_loader:
            assert tuple(x_batch.size()) == (batch_size, lookback, 1)
            assert tuple(y_batch.size()) == (batch_size, horizon, 1)
            break

        # Non-subset relationship
        with pytest.raises(ValueError):
            tsdata.to_torch_data_loader(batch_size=batch_size,
                                        lookback=lookback,
                                        horizon=horizon,
                                        target_col=['value', 'extra feature'])


    @op_torch
    def test_tsdataset_abnormal_check_and_repair(self):
        from bigdl.chronos.data.utils.quality_inspection import _abnormal_value_check
        for val in ['absolute', 'relative']:
            df = get_abnormal_df()
            tsdata = TSDataset.from_pandas(df, dt_col="datetime",
                                           target_col=['a', 'b', 'c', 'd', 'e'],
                                           extra_feature_col=None,
                                           repair=False)
            flag = _abnormal_value_check(tsdata.df, tsdata.dt_col, threshold=3)
            assert flag == False

            if val is 'relative':
                tsdata = tsdata.repair_abnormal_data()
                flag = _abnormal_value_check(tsdata.df, tsdata.dt_col, threshold=3)
                assert flag == True
            else:
                tsdata = tsdata.repair_abnormal_data(mode=val, threshold=(-1, 2))
                flag = _abnormal_value_check(tsdata.df, tsdata.dt_col, threshold=3)
                assert flag == True

    @op_torch
    def test_from_prometheus(self):
        '''
        The selected query str is just for getting data from Prometheus server.
        '''
        import time
        endtime = time.time()
        starttime = endtime - 1000
        prometheus_url = os.getenv("PROMETHEUS_URL")
        tsdata = TSDataset.from_prometheus(prometheus_url=prometheus_url,
                                           query='prometheus_ready',
                                           starttime=starttime,
                                           endtime=endtime,
                                           step="1s")
        assert tsdata.to_pandas().shape[0] == 1001
        assert tsdata.to_pandas().shape[1] == 3
        assert tsdata._id_list == ['0']
        assert tsdata.target_col == ['prometheus_ready'+
                                     '{instance="localhost:9090",job="prometheus"}']
        assert tsdata.dt_col == "datetime"

        tsdata = TSDataset.from_prometheus(prometheus_url=prometheus_url,
                                           query=['prometheus_ready',
                                                  'process_cpu_seconds_total',
                                                  'process_virtual_memory_bytes',
                                                  'process_virtual_memory_max_bytes'],
                                           starttime=starttime,
                                           endtime=endtime,
                                           step="1s",
                                           target_col=['process_cpu_seconds_total{instance'+
                                                       '="localhost:9090",job="prometheus"}',
                                                       'process_virtual_memory_bytes{instance'+
                                                       '="localhost:9090",job="prometheus"}'],
                                           id_col='prometheus_ready{instance'+
                                                  '="localhost:9090",job="prometheus"}',
                                           extra_feature_col='process_virtual_memory_max_bytes'+
                                                             '{instance="localhost:9090",'+
                                                             'job="prometheus"}')
        assert tsdata.to_pandas().shape[0] == 1001
        assert tsdata.to_pandas().shape[1] == 5
        assert tsdata._id_list == [1.0]
        assert tsdata.target_col == ['process_cpu_seconds_total{instance'+
                                     '="localhost:9090",job="prometheus"}',
                                     'process_virtual_memory_bytes{instance'+
                                     '="localhost:9090",job="prometheus"}']
        assert tsdata.feature_col == ['process_virtual_memory_max_bytes'+
                                      '{instance="localhost:9090",job="prometheus"}']
        assert tsdata.dt_col == "datetime"

        # Check the validity of `target_col/id_col/extra_feature_col`
        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_prometheus(prometheus_url=prometheus_url,
                                               query='prometheus_ready',
                                               starttime=starttime,
                                               endtime=endtime,
                                               step="1s",
                                               target_col='process_cpu_seconds_total')

        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_prometheus(prometheus_url=prometheus_url,
                                               query='prometheus_ready',
                                               starttime=starttime,
                                               endtime=endtime,
                                               step="1s",
                                               id_col='process_cpu_seconds_total')

        with pytest.raises(RuntimeError):
            tsdata = TSDataset.from_prometheus(prometheus_url=prometheus_url,
                                               query='prometheus_ready',
                                               starttime=starttime,
                                               endtime=endtime,
                                               step="1s",
                                               extra_feature_col='process_cpu_seconds_total')

    @op_torch
    def test_export_jit(self):
        from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
        import torch

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "inference_data.csv")
        df_single_id = get_ts_df()
        sample_num = len(df_single_id)
        df_single_id["id"] = np.array([0] * sample_num)
        df_multi_id = get_multi_id_ts_df()
        df_multi_id["id"] = np.array([0]*50 + [1]*50)

        lookback = random.randint(1, 20)
        horizon = random.randint(1, 10)

        for df in [df_single_id, df_multi_id]:
            df.to_csv(csv_path, index=False)
            # drop_dt_col=True
            for scaler in [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]:
                tsdata = TSDataset.from_pandas(df,
                                               dt_col="datetime",
                                               target_col="value",
                                               id_col="id",
                                               extra_feature_col="extra feature",
                                               repair=False,
                                               deploy_mode=False)

                tsdata.scale(scaler, fit=True)\
                      .roll(lookback=lookback, horizon=horizon)

                tsdata.export_jit(path_dir=temp_dir, drop_dt_col=True)

                # deployment
                deployment_df = pd.read_csv(csv_path, parse_dates=["datetime"])
                tsdata = TSDataset.from_pandas(deployment_df,
                                               dt_col="datetime",
                                               target_col="value",
                                               id_col="id",
                                               extra_feature_col="extra feature",
                                               repair=False,
                                               deploy_mode=True)
                # drop datetime col since it can't be converted to Pytorch tensor
                tsdata.df.drop(columns=tsdata.dt_col, inplace=True)
                input_tensor = torch.from_numpy(tsdata.df.values).type(torch.float64)
                preprocess_path = os.path.join(temp_dir, "tsdata_preprocessing.pt")
                preprocess_module = torch.jit.load(preprocess_path)
                preprocess_output = preprocess_module.forward(input_tensor)

                tsdata.scale(scaler=scaler, fit=False)\
                      .roll(lookback=lookback, horizon=horizon)

                assert_array_almost_equal(tsdata.to_numpy(), preprocess_output.numpy())

                postprocess_path = os.path.join(temp_dir, "tsdata_postprocessing.pt")
                postprocess_module = torch.jit.load(postprocess_path)
                postprocess_output = postprocess_module.forward(preprocess_output)

                unscale_data = tsdata.unscale_numpy(tsdata.to_numpy())
                assert_array_almost_equal(unscale_data, postprocess_output.numpy())

            # drop_dt_col=False
            for scaler in [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]:
                tsdata = TSDataset.from_pandas(df,
                                               dt_col="datetime",
                                               target_col="value",
                                               id_col="id",
                                               extra_feature_col="extra feature",
                                               repair=False,
                                               deploy_mode=False)

                tsdata.scale(scaler, fit=True)\
                      .roll(lookback=lookback, horizon=horizon)

                preprocess_module, postprocess_module = tsdata.export_jit(drop_dt_col=False)
                preprocess_path = os.path.join(temp_dir, "preprocess_module.pt")
                torch.jit.save(preprocess_module, preprocess_path)
                postprocess_path = os.path.join(temp_dir, "postprocess_module.pt")
                torch.jit.save(postprocess_module, postprocess_path)

                # deployment
                deployment_df = pd.read_csv(csv_path, parse_dates=["datetime"])
                tsdata = TSDataset.from_pandas(deployment_df,
                                               dt_col="datetime",
                                               target_col="value",
                                               id_col="id",
                                               extra_feature_col="extra feature",
                                               repair=False,
                                               deploy_mode=True)
                # give datetime an integer value to convert the dataframe to Pytorch tensor
                # since datetime col will not be used in scale and roll, so we can give it
                # a meaningless value
                tsdata.df["datetime"] = np.array([1000]*len(tsdata.df))
                input_tensor = torch.from_numpy(tsdata.df.values).type(torch.float64)
                preprocess = torch.jit.load(preprocess_path)
                preprocess_output = preprocess.forward(input_tensor)

                tsdata.scale(scaler=scaler, fit=False)\
                      .roll(lookback=lookback, horizon=horizon)

                assert_array_almost_equal(tsdata.to_numpy(), preprocess_output.numpy())

                postprocess_module = torch.jit.load(postprocess_path)
                postprocess_output = postprocess_module.forward(preprocess_output)

                unscale_data = tsdata.unscale_numpy(tsdata.to_numpy())
                assert_array_almost_equal(unscale_data, postprocess_output.numpy())

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
