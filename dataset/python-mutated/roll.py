import numpy as np
import pandas as pd

def roll_timeseries_dataframe(df, roll_feature_df, lookback, horizon, feature_col, target_col, id_col=None, label_len=0, contain_id=False, deploy_mode=False):
    if False:
        print('Hello World!')
    '\n    roll dataframe into numpy ndarray sequence samples.\n\n    :param input_df: a dataframe which has been resampled in uniform frequency.\n    :param roll_feature_df: an additional rolling feature dataframe that will\n           be append to final result.\n    :param lookback: the length of the past sequence\n    :param horizon: int or list,\n           if `horizon` is an int, we will sample `horizon` step\n           continuously after the forecasting point.\n           if `horizon` is an list, we will sample discretely according\n           to the input list. 1 means the timestamp just after the observed data.\n    :param feature_col: list, indicate the feature col name.\n    :param target_col: list, indicate the target col name.\n    :param id_col: str, indicate the id col name, only needed when contain_id is True.\n    :param label_len: This parameter is only for transformer-based model.\n    :param contain_id: This parameter is only for XShardsTSDataset\n    :param deploy_mode: a bool indicates whether to use deploy mode, which will be used in\n           production environment to reduce the latency of data processing. The value\n           defaults to False.\n    :return: x, y\n        x: 3-d numpy array in format (no. of samples, lookback, feature_col length)\n        y: 3-d numpy array in format (no. of samples, horizon, target_col length)\n    Note: Specially, if `horizon` is set to 0, then y will be None.\n    '
    if deploy_mode:
        return _roll_timeseries_dataframe_test(df, roll_feature_df, lookback, feature_col, target_col, id_col=id_col, contain_id=contain_id)
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(isinstance(df, pd.DataFrame), 'df is expected to be pandas dataframe')
    invalidInputError(isinstance(lookback, int), 'lookback is expected to be int')
    invalidInputError(isinstance(feature_col, list), 'feature_col is expected to be list')
    invalidInputError(isinstance(target_col, list), 'target_col is expected to be list')
    is_horizon_int = isinstance(horizon, int)
    is_horizon_list = isinstance(horizon, list) and isinstance(horizon[0], int) and (min(horizon) > 0)
    invalidInputError(is_horizon_int or is_horizon_list, 'horizon is expected to be a list or int')
    is_test = True if is_horizon_int and horizon == 0 and (label_len == 0) else False
    if not is_test:
        return _roll_timeseries_dataframe_train(df, roll_feature_df, lookback, horizon, feature_col, target_col, id_col=id_col, label_len=label_len, contain_id=contain_id)
    else:
        return _roll_timeseries_dataframe_test(df, roll_feature_df, lookback, feature_col, target_col, id_col=id_col, contain_id=contain_id)

def _append_rolling_feature_df(rolling_result, roll_feature_df):
    if False:
        while True:
            i = 10
    if roll_feature_df is None:
        return rolling_result
    additional_rolling_result = np.zeros((rolling_result.shape[0], rolling_result.shape[1], len(roll_feature_df.columns)))
    for idx in range(additional_rolling_result.shape[0]):
        for col_idx in range(additional_rolling_result.shape[2]):
            additional_rolling_result[idx, :, col_idx] = roll_feature_df.iloc[idx, col_idx]
    rolling_result = np.concatenate([rolling_result, additional_rolling_result], axis=2)
    return rolling_result

def _roll_timeseries_dataframe_test(df, roll_feature_df, lookback, feature_col, target_col, id_col, contain_id):
    if False:
        for i in range(10):
            print('nop')
    x = df.loc[:, target_col + feature_col].values.astype(np.float32)
    (output_x, mask_x) = _roll_timeseries_ndarray(x, lookback)
    mask = mask_x == 1
    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)
    if contain_id:
        return (x, None, df.loc[:, [id_col]].values)
    else:
        return (x, None)

def _roll_timeseries_dataframe_train(df, roll_feature_df, lookback, horizon, feature_col, target_col, id_col, label_len, contain_id):
    if False:
        return 10
    from bigdl.nano.utils.common import invalidInputError
    if label_len != 0 and isinstance(horizon, list):
        invalidInputError(False, 'horizon should be an integer if label_len is set to larger than 0.')
    max_horizon = horizon if isinstance(horizon, int) else max(horizon)
    if max_horizon > 0:
        x = df[:-max_horizon].loc[:, target_col + feature_col].values.astype(np.float32)
    else:
        x = df.loc[:, target_col + feature_col].values.astype(np.float32)
    y = df.iloc[lookback - label_len:].loc[:, target_col].values.astype(np.float32)
    (output_x, mask_x) = _roll_timeseries_ndarray(x, lookback)
    if isinstance(horizon, list):
        (output_y, mask_y) = _roll_timeseries_ndarray(y, horizon)
    else:
        (output_y, mask_y) = _roll_timeseries_ndarray(y, horizon + label_len)
    mask = (mask_x == 1) & (mask_y == 1)
    x = _append_rolling_feature_df(output_x[mask], roll_feature_df)
    if contain_id:
        return (x, output_y[mask], df.loc[:, [id_col]].values)
    else:
        return (x, output_y[mask])

def _shift(arr, num, fill_value=np.nan):
    if False:
        return 10
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def _roll_timeseries_ndarray(data, window):
    if False:
        while True:
            i = 10
    '\n    data should be a ndarray with num_dim = 2\n    first dim is timestamp\n    second dim is feature\n    '
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(data.ndim == 2, 'data dim is expected to be 2')
    data = np.expand_dims(data, axis=1)
    window_size = window if isinstance(window, int) else max(window)
    if isinstance(window, int):
        window_idx = np.arange(window)
    else:
        window_idx = np.array(window) - 1
    roll_data = np.concatenate([_shift(data, i) for i in range(0, -window_size, -1)], axis=1)
    if data.shape[0] >= window_size:
        roll_data = roll_data[:data.shape[0] - window_size + 1, window_idx, :]
    else:
        roll_data = roll_data[:0, window_idx, :]
    mask = ~np.any(np.isnan(roll_data), axis=(1, 2))
    return (roll_data, mask)