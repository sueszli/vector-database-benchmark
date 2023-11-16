import pytest
import pandas as pd
import numpy as np
from unittest import TestCase
from bigdl.chronos.data.utils.quality_inspection import quality_check_timeseries_dataframe
from ... import op_torch, op_tf2

def get_missing_df():
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
    df['a'] = np.nan
    df['datetime'] = pd.date_range('1/1/2019', periods=50)
    return df

def get_multi_interval_df():
    if False:
        i = 10
        return i + 15
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['datetime'] = pd.date_range('1/1/2019', periods=50)
    df['datetime'][25:] = pd.date_range('1/1/2020', periods=25)
    return df

def get_non_dt_df():
    if False:
        while True:
            i = 10
    data = np.random.random_sample((50, 5))
    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
    df['datetime'] = ['2022-1-1'] * 50
    return df

@op_torch
@op_tf2
class TestCheckAndRepairTimeSeries(TestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        pass

    def teardown_method(self, method):
        if False:
            return 10
        pass

    def test_normal_dataframe(self):
        if False:
            while True:
                i = 10
        pass

    def test_missing_check_and_repair(self):
        if False:
            while True:
                i = 10
        df = get_missing_df()
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is False
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=True)
        assert flag is True
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is True

    def test_time_interval_check_and_repair(self):
        if False:
            for i in range(10):
                print('nop')
        df = get_multi_interval_df()
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is False
        (flag, df) = quality_check_timeseries_dataframe(df, 'datetime', repair=True)
        assert flag is True
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is True

    def test_non_dt_type_check_and_repair(self):
        if False:
            i = 10
            return i + 15
        df = get_non_dt_df()
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is False
        (flag, df) = quality_check_timeseries_dataframe(df, 'datetime', repair=True)
        assert flag is True
        (flag, _) = quality_check_timeseries_dataframe(df, 'datetime', repair=False)
        assert flag is True