from typing import List
import numpy as np
import pandas as pd
from pandas import Timedelta
from pandas.tseries.frequencies import to_offset
from bigdl.chronos.data.utils.roll import _roll_timeseries_ndarray

class TimeFeature:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        pass

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '()'

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            return 10
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            return 10
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            return 10
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            while True:
                i = 10
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            return 10
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        if False:
            while True:
                i = 10
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(offset) -> List[TimeFeature]:
    if False:
        while True:
            i = 10
    '\n    Returns a list of time features that will be appropriate for the given frequency string.\n    Parameters\n    ----------\n    freq_str\n        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.\n    '
    features_by_offsets = ((Timedelta(seconds=60), [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]), (Timedelta(minutes=60), [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]), (Timedelta(hours=24), [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]), (Timedelta(days=7), [DayOfWeek, DayOfMonth, DayOfYear]), (Timedelta(days=30), [DayOfMonth, WeekOfYear]), (Timedelta(days=365), [MonthOfYear]))
    for (offset_type, feature_classes) in features_by_offsets:
        if offset < offset_type:
            return [cls() for cls in feature_classes]
    return []

def time_features(dates, freq='h'):
    if False:
        print('Hello World!')
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

def gen_time_enc_arr(df, dt_col, freq, horizon_time, is_predict, lookback, label_len):
    if False:
        i = 10
        return i + 15
    df_stamp = pd.DataFrame(columns=[dt_col])
    if is_predict:
        pred_dates = pd.date_range(df[dt_col].values[-1], periods=horizon_time + 1, freq=freq)
        df_stamp.loc[:, dt_col] = list(df[dt_col].values) + list(pred_dates[1:])
    else:
        df_stamp.loc[:, dt_col] = list(df[dt_col].values)
    data_stamp = time_features(pd.to_datetime(df_stamp[dt_col].values), freq=freq)
    data_stamp = data_stamp.transpose(1, 0)
    max_horizon = horizon_time if isinstance(horizon_time, int) else max(horizon_time)
    (numpy_x_timeenc, _) = _roll_timeseries_ndarray(data_stamp[:-max_horizon], lookback)
    (numpy_y_timeenc, _) = _roll_timeseries_ndarray(data_stamp[lookback - label_len:], horizon_time + label_len)
    return (numpy_x_timeenc, numpy_y_timeenc)