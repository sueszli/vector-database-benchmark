from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def make_temporal_cutoffs(instance_ids, cutoffs, window_size=None, num_windows=None, start=None):
    if False:
        return 10
    'Makes a set of equally spaced cutoff times prior to a set of input cutoffs and instance ids.\n\n    If window_size and num_windows are provided, then num_windows of size window_size will be created\n    prior to each cutoff time\n\n    If window_size and a start list is provided, then a variable number of windows will be created prior\n    to each cutoff time, with the corresponding start time as the first cutoff.\n\n    If num_windows and a start list is provided, then num_windows of variable size will be created prior\n    to each cutoff time, with the corresponding start time as the first cutoff\n\n    Args:\n        instance_ids (list, np.ndarray, or pd.Series): list of instance ids. This function will make a\n            new datetime series of multiple cutoff times for each value in this array.\n        cutoffs (list, np.ndarray, or pd.Series): list of datetime objects associated with each instance id.\n            Each one of these will be the last time in the new datetime series for each instance id\n        window_size (pd.Timedelta, optional): amount of time between each datetime in each new cutoff series\n        num_windows (int, optional): number of windows in each new cutoff series\n        start (list, optional): list of start times for each instance id\n    '
    if window_size is not None and num_windows is not None and (start is not None):
        raise ValueError('Only supply 2 of the 3 optional args, window_size, num_windows and start')
    out = []
    for (i, id_time) in enumerate(zip(instance_ids, cutoffs)):
        (_id, time) = id_time
        _window_size = window_size
        _start = None
        if start is not None:
            if window_size is None:
                _window_size = (time - start[i]) / (num_windows - 1)
            else:
                _start = start[i]
        to_add = pd.DataFrame()
        to_add['time'] = pd.date_range(end=time, periods=num_windows, freq=_window_size, start=_start)
        to_add['instance_id'] = [_id] * len(to_add['time'])
        out.append(to_add)
    return pd.concat(out).reset_index(drop=True)

def convert_time_units(secs, unit):
    if False:
        while True:
            i = 10
    '\n    Converts a time specified in seconds to a time in the given units\n\n    Args:\n        secs (integer): number of seconds. This function will convert the units of this number.\n        unit(str): units to be converted to.\n            acceptable values: years, months, days, hours, minutes, seconds, milliseconds, nanoseconds\n    '
    unit_divs = {'years': 31540000, 'months': 2628000, 'days': 86400, 'hours': 3600, 'minutes': 60, 'seconds': 1, 'milliseconds': 0.001, 'nanoseconds': 1e-09}
    if unit not in unit_divs:
        raise ValueError('Invalid unit given, make sure it is plural')
    return secs / unit_divs[unit]

def convert_datetime_to_floats(x):
    if False:
        for i in range(10):
            print('nop')
    first = int(x.iloc[0].value * 1e-09)
    x = pd.to_numeric(x).astype(np.float64).values
    dividend = find_dividend_by_unit(first)
    x *= 1e-09 / dividend
    return x

def convert_timedelta_to_floats(x):
    if False:
        while True:
            i = 10
    first = int(x.iloc[0].total_seconds())
    dividend = find_dividend_by_unit(first)
    x = pd.TimedeltaIndex(x).total_seconds().astype(np.float64) / dividend
    return x

def find_dividend_by_unit(time):
    if False:
        while True:
            i = 10
    'Finds whether time best corresponds to a value in\n    days, hours, minutes, or seconds.\n    '
    for dividend in [86400, 3600, 60]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1

def calculate_trend(series):
    if False:
        for i in range(10):
            print('nop')
    if series.dtype == 'Int64':
        series = series.astype('float64')
    df = pd.DataFrame({'x': series.index, 'y': series.values}).dropna()
    if df.shape[0] <= 2:
        return np.nan
    if isinstance(df['x'].iloc[0], (datetime, pd.Timestamp)):
        x = convert_datetime_to_floats(df['x'])
    else:
        x = df['x'].values
    if isinstance(df['y'].iloc[0], (datetime, pd.Timestamp)):
        y = convert_datetime_to_floats(df['y'])
    elif isinstance(df['y'].iloc[0], (timedelta, pd.Timedelta)):
        y = convert_timedelta_to_floats(df['y'])
    else:
        y = df['y'].values
    x = x - x.mean()
    y = y - y.mean()
    if len(np.unique(x)) == 1:
        return 0
    coefficients = np.polyfit(x, y, 1)
    return coefficients[0]