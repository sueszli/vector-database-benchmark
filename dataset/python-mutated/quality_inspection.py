import pandas as pd
import numpy as np
from bigdl.nano.utils.common import invalidInputError
import logging

def quality_check_timeseries_dataframe(df, dt_col, id_col=None, repair=True):
    if False:
        i = 10
        return i + 15
    '\n    detect the low-quality data and provide suggestion (e.g. call .impute or .resample).\n\n    :param df: a pandas dataframe for your raw time series data.\n    :param dt_col: a str indicates the col name of datetime\n           column in the input data frame, the dt_col must be sorted\n           from past to latest respectively for each id.\n    :param id_col: (optional) a str indicates the col name of dataframe id. If\n           it is not explicitly stated, then the data is interpreted as only\n           containing a single id.\n    :param repair: a bool indicates whether automaticly repair low quality data.\n\n    :return: a bool indicates df whether contains low-quality data.\n    '
    invalidInputError(dt_col in df.columns, f'dt_col {dt_col} can not be found in df.')
    if id_col is not None:
        invalidInputError(id_col in df.columns, f'id_col {id_col} can not be found in df.')
    invalidInputError(pd.isna(df[dt_col]).sum() == 0, 'There is N/A in datetime col')
    if df.empty is True:
        return (True, df)
    flag = True
    if _timestamp_type_check(df[dt_col]) is False:
        if repair is True:
            flag = flag and _timestamp_type_repair(df, dt_col)
        else:
            flag = False
    if flag is True:
        (interval_flag, intervals) = _time_interval_check(df, dt_col, id_col)
        if interval_flag is False:
            if repair is True:
                (df, repair_flag) = _time_interval_repair(df, dt_col, intervals, id_col)
                flag = flag and repair_flag
            else:
                flag = False
    if _missing_value_check(df, dt_col) is False:
        if repair is True:
            flag = flag and _missing_value_repair(df, dt_col)
        else:
            flag = False
    _abnormal_value_check(df, dt_col)
    return (flag, df)

def _timestamp_type_check(df_column):
    if False:
        i = 10
        return i + 15
    '\n    This check is used to make datetime column is datetime64 stype to facilitate our\n    access to freq.\n    '
    _is_pd_datetime = pd.api.types.is_datetime64_any_dtype(df_column.dtypes)
    if _is_pd_datetime is not True:
        logging.warning('Datetime column should be datetime64 dtype. You can manually modify the dtype, or set repair=True when initialize TSDataset.')
        return False
    return True

def _timestamp_type_repair(df, dt_col):
    if False:
        i = 10
        return i + 15
    '\n    This repair is used to convert object or other non datetime64 timestamp column\n    to datetime dtype.\n    '
    try:
        df[dt_col] = df[dt_col].astype('datetime64')
    except:
        return False
    logging.warning('Datetime column has be modified to datetime64 dtype.')
    return True

def _time_interval_check(df, dt_col, id_col=None):
    if False:
        print('Hello World!')
    '\n    This check is used to verify whether all the time intervals of datetime column\n    are consistent.\n    '
    if id_col is not None:
        _id_list = df[id_col].unique()
    if id_col is not None and len(_id_list) > 1:
        flag = True

        def get_interval(x):
            if False:
                print('Hello World!')
            df_column = x[dt_col]
            interval = df_column.shift(-1) - df_column
            unique_intervals = interval[:-1].unique()
            return unique_intervals
        group = df.groupby(id_col).apply(get_interval)
        for ind in group.index:
            unique_intervals = group[ind]
            if len(unique_intervals) > 1:
                flag = False
        if flag is True:
            return (True, None)
        else:
            logging.warning('There are irregular interval(more than one interval length) among the data. You can call .resample(interval).impute() first to clean the data manually, or set repair=True when initialize TSDataset.')
            return (False, None)
    else:
        df_column = df[dt_col]
        intervals = df_column.shift(-1) - df_column
        unique_intervals = intervals[:-1].unique()
        if len(unique_intervals) > 1:
            logging.warning('There are irregular interval(more than one interval length) among the data. You can call .resample(interval).impute() first to clean the data manually, or set repair=True when initialize TSDataset.')
            return (False, intervals)
        return (True, intervals)

def _time_interval_repair(df, dt_col, intervals, id_col=None):
    if False:
        return 10
    '\n    This check is used to get consitent time interval by resample data according to\n    the mode of original intervals.\n    '
    if id_col is not None and intervals is None:
        from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
        try:

            def resample_interval(x):
                if False:
                    for i in range(10):
                        print('nop')
                df_column = x[dt_col]
                interval = df_column.shift(-1) - df_column
                intervals = interval[:-1]
                mode = intervals.mode()[0]
                df = resample_timeseries_dataframe(x, dt_col=dt_col, interval=mode, id_col=id_col)
                return df
            new_df = df.groupby(id_col, as_index=False).apply(resample_interval)
            new_df.reset_index(drop=True, inplace=True)
            logging.warning('Dataframe has be resampled.')
            return (new_df, True)
        except:
            return (df, False)
    else:
        mode = intervals[:-1].mode()[0]
        from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
        try:
            df = resample_timeseries_dataframe(df, dt_col=dt_col, interval=mode, id_col=id_col)
            logging.warning(f'Dataframe has be resampled according to interval {mode}.')
            return (df, True)
        except:
            return (df, False)

def _missing_value_check(df, dt_col, threshold=0):
    if False:
        return 10
    '\n    This check is used to determine whether there are missing values in the data.\n    '
    for column in df.columns:
        if column == dt_col:
            continue
        df_col = df[column]
        missing_value = df_col.isna().sum()
        rows = len(df)
        if missing_value / rows > threshold:
            logging.warning(f'The missing value of column {column} exceeds {threshold},please call .impute() fisrt to remove N/A number manually, or set repair=True when initialize TSDataset.')
            return False
    return True

def _missing_value_repair(df, dt_col):
    if False:
        while True:
            i = 10
    '\n    This repair is used to fill missing value with impute by linear interpolation.\n    '
    try:
        temp_col = df[dt_col]
        df[dt_col] = 0
        df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
        df[dt_col] = temp_col
        df.fillna(0, inplace=True)
    except:
        return False
    logging.warning('Missing data has be imputed.')
    return True

def _abnormal_value_check(df, dt_col, threshold=10):
    if False:
        while True:
            i = 10
    '\n    This check is used to determine whether there are abnormal values in the data.\n    '
    for column in df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(df[column]):
            continue
        df_col = df[column]
        std_val = df_col.std()
        mean_val = df_col.mean()
        df_col = df_col.apply(lambda x: x - mean_val)
        if df_col.max() > std_val * threshold or df_col.min() < -std_val * threshold:
            logging.warning(f'Some values of column {column} exceeds the mean plus/minus {threshold} times standard deviation, please call .repair_abnormal_data() to remove abnormal values.')
            return False
    return True

def _abnormal_value_repair(df, dt_col, mode, threshold):
    if False:
        return 10
    '\n    This repair is used to replace detected abnormal data with the last non N/A number.\n    '
    invalidInputError(mode in ['absolute', 'relative'], f"mode should be one of ['absolute', 'relative'], but found {mode}.")
    if mode == 'absolute':
        invalidInputError(isinstance(threshold, tuple), f"threshold should be a tuple when mode is set to 'absolute', but found {type(threshold)}.")
        invalidInputError(threshold[0] <= threshold[1], f"threshold should be a tuple (min_value, max_value) when mode is set to 'absolute', but found {threshold}.")
        res_df = _abs_abnormal_value_repair(df, dt_col, threshold)
    else:
        invalidInputError(isinstance(threshold, float), f"threshold should be a float when mode is set to 'relative', but found {type(threshold)}.")
        res_df = _rel_abnormal_value_repair(df, dt_col, threshold)
    return res_df

def _abs_abnormal_value_repair(df, dt_col, threshold):
    if False:
        return 10
    res_df = df.copy()
    for column in res_df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(res_df[column]):
            continue
        res_df[column] = res_df[column].apply(lambda x: np.nan if x < threshold[0] or x > threshold[1] else x)
    res_df.iloc[0] = res_df.iloc[0].fillna(0)
    res_df = res_df.fillna(method='pad')
    return res_df

def _rel_abnormal_value_repair(df, dt_col, threshold):
    if False:
        print('Hello World!')
    res_df = df.copy()
    for column in res_df.columns:
        if column == dt_col or pd.api.types.is_string_dtype(res_df[column]):
            continue
        std_val = res_df[column].std()
        mean_val = res_df[column].mean()
        res_df[column] = res_df[column].apply(lambda x: np.nan if x > mean_val + threshold * std_val or x < mean_val - threshold * std_val else x)
    res_df.iloc[0] = res_df.iloc[0].fillna(0)
    res_df = res_df.fillna(method='pad')
    return res_df