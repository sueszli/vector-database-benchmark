from __future__ import annotations
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from neuralprophet.configure import ConfigEvents, ConfigLaggedRegressors, ConfigSeasonality
log = logging.getLogger('NP.df_utils')

@dataclass
class ShiftScale:
    shift: float = 0.0
    scale: float = 1.0

def prep_or_copy_df(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool, list[str]]:
    if False:
        for i in range(10):
            print('nop')
    "Copy df if it contains the ID column. Creates ID column with '__df__' if it is a df with a single time series.\n    Parameters\n    ----------\n        df : pd.DataFrame\n            df or dict containing data\n    Returns\n    -------\n        pd.DataFrames\n            df with ID col\n        bool\n            whether the ID col was present\n        bool\n            wheter it is a single time series\n        list\n            list of IDs\n    "
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Provided DataFrame (df) must be of pd.DataFrame type.')
    df_copy = df.copy(deep=True)
    df_has_id_column = 'ID' in df_copy.columns
    if not df_has_id_column:
        log.debug('Provided DataFrame (df) contains a single time series.')
        df_copy['ID'] = '__df__'
        return (df_copy, df_has_id_column, True, ['__df__'])
    unique_id_values = list(df_copy['ID'].unique())
    df_has_single_time_series = len(unique_id_values) == 1
    single_or_multiple_message = 'a single' if df_has_single_time_series else 'multiple'
    log.debug(f'Provided DataFrame (df) has an ID column and contains {single_or_multiple_message} time series.')
    return (df_copy, df_has_id_column, df_has_single_time_series, unique_id_values)

def return_df_in_original_format(df, received_ID_col=False, received_single_time_series=True):
    if False:
        for i in range(10):
            print('nop')
    'Return dataframe in the original format.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            df with data\n        received_ID_col : bool\n            whether the ID col was present\n        received_single_time_series: bool\n            wheter it is a single time series\n    Returns\n    -------\n        pd.Dataframe\n            original input format\n    '
    new_df = df.copy(deep=True)
    if not received_ID_col and received_single_time_series:
        assert len(new_df['ID'].unique()) == 1
        new_df.drop('ID', axis=1, inplace=True)
        log.info('Returning df with no ID column')
    return new_df

def get_max_num_lags(config_lagged_regressors: Optional[ConfigLaggedRegressors], n_lags: int) -> int:
    if False:
        while True:
            i = 10
    'Get the greatest number of lags between the autoregression lags and the covariates lags.\n\n    Parameters\n    ----------\n        config_lagged_regressors : configure.ConfigLaggedRegressors\n            Configurations for lagged regressors\n        n_lags : int\n            number of lagged values of series to include as model inputs\n\n    Returns\n    -------\n        int\n            Maximum number of lags between the autoregression lags and the covariates lags.\n    '
    if config_lagged_regressors is not None:
        log.debug('config_lagged_regressors exists')
        max_n_lags = max([n_lags] + [val.n_lags for (key, val) in config_lagged_regressors.items()])
    else:
        log.debug('config_lagged_regressors does not exist')
        max_n_lags = n_lags
    return max_n_lags

def merge_dataframes(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    "Join dataframes for procedures such as splitting data, set auto seasonalities, and others.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            containing column ``ds``, ``y``, and ``ID`` with data\n\n    Returns\n    -------\n        pd.Dataframe\n            Dataframe with concatenated time series (sorted 'ds', duplicates removed, index reset)\n    "
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Can not join other than pd.DataFrames')
    if 'ID' not in df.columns:
        raise ValueError("df does not contain 'ID' column")
    df_merged = df.copy(deep=True).drop('ID', axis=1)
    df_merged = df_merged.sort_values('ds')
    df_merged = df_merged.drop_duplicates(subset=['ds'])
    df_merged = df_merged.reset_index(drop=True)
    return df_merged

def data_params_definition(df, normalize, config_lagged_regressors: Optional[ConfigLaggedRegressors]=None, config_regressors=None, config_events: Optional[ConfigEvents]=None, config_seasonality: Optional[ConfigSeasonality]=None, local_run_despite_global: Optional[bool]=None):
    if False:
        while True:
            i = 10
    '\n    Initialize data scaling values.\n\n    Note\n    ----\n    We do a z normalization on the target series ``y``,\n    unlike OG Prophet, which does shift by min and scale by max.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        Time series to compute normalization parameters from.\n    normalize : str\n        Type of normalization to apply to the time series.\n\n            options:\n\n                ``soft`` (default), unless the time series is binary, in which case ``minmax`` is applied.\n\n                ``off`` bypasses data normalization\n\n                ``minmax`` scales the minimum value to 0.0 and the maximum value to 1.0\n\n                ``standardize`` zero-centers and divides by the standard deviation\n\n                ``soft`` scales the minimum value to 0.0 and the 95th quantile to 1.0\n\n                ``soft1`` scales the minimum value to 0.1 and the 90th quantile to 0.9\n    config_lagged_regressors : configure.ConfigLaggedRegressors\n        Configurations for lagged regressors\n    normalize : bool\n        data normalization\n    config_regressors : configure.ConfigFutureRegressors\n        extra regressors (with known future values) with sub_parameters normalize (bool)\n    config_events : configure.ConfigEvents\n        user specified events configs\n    config_seasonality : configure.ConfigSeasonality\n        user specified seasonality configs\n\n    Returns\n    -------\n    OrderedDict\n        scaling values with ShiftScale entries containing ``shift`` and ``scale`` parameters.\n    '
    data_params = OrderedDict({})
    if df['ds'].dtype == np.int64:
        df['ds'] = df.loc[:, 'ds'].astype(str)
    df['ds'] = pd.to_datetime(df.loc[:, 'ds'])
    data_params['ds'] = ShiftScale(shift=df['ds'].min(), scale=df['ds'].max() - df['ds'].min())
    if 'y' in df:
        data_params['y'] = get_normalization_params(array=df['y'].values, norm_type=normalize)
    if config_lagged_regressors is not None:
        for covar in config_lagged_regressors.keys():
            if covar not in df.columns:
                raise ValueError(f'Lagged regressor {covar} not found in DataFrame.')
            norm_type_lag = config_lagged_regressors[covar].normalize
            if local_run_despite_global:
                if len(df[covar].unique()) < 2:
                    norm_type_lag = 'soft'
            data_params[covar] = get_normalization_params(array=df[covar].values, norm_type=norm_type_lag)
    if config_regressors is not None:
        for reg in config_regressors.keys():
            if reg not in df.columns:
                raise ValueError(f'Regressor {reg} not found in DataFrame.')
            norm_type = config_regressors[reg].normalize
            if local_run_despite_global:
                if len(df[reg].unique()) < 2:
                    norm_type = 'soft'
            data_params[reg] = get_normalization_params(array=df[reg].values, norm_type=norm_type)
    if config_events is not None:
        for event in config_events.keys():
            if event not in df.columns:
                raise ValueError(f'Event {event} not found in DataFrame.')
            data_params[event] = ShiftScale()
    if config_seasonality is not None:
        for season in config_seasonality.periods:
            condition_name = config_seasonality.periods[season].condition_name
            if condition_name is not None:
                if condition_name not in df.columns:
                    raise ValueError(f'Seasonality condition {condition_name} not found in DataFrame.')
                data_params[condition_name] = ShiftScale()
    return data_params

def init_data_params(df, normalize='auto', config_lagged_regressors: Optional[ConfigLaggedRegressors]=None, config_regressors=None, config_events: Optional[ConfigEvents]=None, config_seasonality: Optional[ConfigSeasonality]=None, global_normalization=False, global_time_normalization=False):
    if False:
        return 10
    'Initialize data scaling values.\n\n    Note\n    ----\n    We compute and store local and global normalization parameters independent of settings.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data to compute normalization parameters from.\n        normalize : str\n            Type of normalization to apply to the time series.\n\n                options:\n\n                    ``soft`` (default), unless the time series is binary, in which case ``minmax`` is applied.\n\n                    ``off`` bypasses data normalization\n\n                    ``minmax`` scales the minimum value to 0.0 and the maximum value to 1.0\n\n                    ``standardize`` zero-centers and divides by the standard deviation\n\n                    ``soft`` scales the minimum value to 0.0 and the 95th quantile to 1.0\n\n                    ``soft1`` scales the minimum value to 0.1 and the 90th quantile to 0.9\n        config_lagged_regressors : configure.ConfigLaggedRegressors\n            Configurations for lagged regressors\n        config_regressors : configure.ConfigFutureRegressors\n            extra regressors (with known future values)\n        config_events : configure.ConfigEvents\n            user specified events configs\n        config_seasonality : configure.ConfigSeasonality\n            user specified seasonality configs\n        global_normalization : bool\n\n            ``True``: sets global modeling training with global normalization\n\n            ``False``: sets global modeling training with local normalization\n        global_time_normalization : bool\n\n            ``True``: normalize time globally across all time series\n\n            ``False``: normalize time locally for each time series\n\n            (only valid in case of global modeling - local normalization)\n\n    Returns\n    -------\n        OrderedDict\n            nested dict with data_params for each dataset where each contains\n        OrderedDict\n            ShiftScale entries containing ``shift`` and ``scale`` parameters for each column\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    df_merged = df.copy(deep=True).drop('ID', axis=1)
    global_data_params = data_params_definition(df_merged, normalize, config_lagged_regressors, config_regressors, config_events, config_seasonality)
    if global_normalization:
        log.debug(f'Global Normalization Data Parameters (shift, scale): {[(k, v) for (k, v) in global_data_params.items()]}')
    local_data_params = OrderedDict()
    local_run_despite_global = True if global_normalization else None
    for (df_name, df_i) in df.groupby('ID'):
        df_i.drop('ID', axis=1, inplace=True)
        local_data_params[df_name] = data_params_definition(df_i, normalize, config_lagged_regressors, config_regressors, config_events, config_seasonality, local_run_despite_global)
        if global_time_normalization:
            local_data_params[df_name]['ds'] = global_data_params['ds']
        if not global_normalization:
            params = [(k, v) for (k, v) in local_data_params[df_name].items()]
            log.debug(f'Local Normalization Data Parameters (shift, scale): {params}')
    return (local_data_params, global_data_params)

def auto_normalization_setting(array):
    if False:
        return 10
    if len(np.unique(array)) < 2:
        raise ValueError('Encountered variable with singular value in training set. Please remove variable.')
    elif len(np.unique(array)) == 2:
        return 'minmax'
    else:
        return 'soft'

def get_normalization_params(array, norm_type):
    if False:
        i = 10
        return i + 15
    if norm_type == 'auto':
        norm_type = auto_normalization_setting(array)
    shift = 0.0
    scale = 1.0
    non_nan_array = array[~np.isnan(array)]
    if norm_type == 'soft':
        lowest = np.min(non_nan_array)
        q95 = np.quantile(non_nan_array, 0.95)
        width = q95 - lowest
        if math.isclose(width, 0):
            width = np.max(non_nan_array) - lowest
        shift = lowest
        scale = width
    elif norm_type == 'soft1':
        lowest = np.min(non_nan_array)
        q90 = np.quantile(non_nan_array, 0.9)
        width = q90 - lowest
        if math.isclose(width, 0):
            width = (np.max(non_nan_array) - lowest) / 1.25
        shift = lowest - 0.125 * width
        scale = 1.25 * width
    elif norm_type == 'minmax':
        shift = np.min(non_nan_array)
        scale = np.max(non_nan_array) - shift
    elif norm_type == 'standardize':
        shift = np.mean(non_nan_array)
        scale: float = np.std(non_nan_array)
    elif norm_type != 'off':
        log.error(f'Normalization {norm_type} not defined.')
    return ShiftScale(shift, scale)

def normalize(df, data_params):
    if False:
        i = 10
        return i + 15
    '\n    Applies data scaling factors to df using data_params.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            with columns ``ds``, ``y``, (and potentially more regressors)\n        data_params : OrderedDict\n            scaling values, as returned by init_data_params with ShiftScale entries containing ``shift`` and ``scale``\n            parameters\n\n    Returns\n    -------\n        pd.DataFrame\n            normalized dataframes\n    '
    df = df.copy(deep=True)
    for name in df.columns:
        if name not in data_params.keys():
            raise ValueError(f'Unexpected column {name} in data')
        new_name = name
        if name == 'ds':
            new_name = 't'
        if name == 'y':
            new_name = 'y_scaled'
        df[new_name] = df[name].sub(data_params[name].shift).div(data_params[name].scale)
    return df

def check_dataframe(df: pd.DataFrame, check_y: bool=True, covariates=None, regressors=None, events=None, seasonalities=None, future: Optional[bool]=None) -> Tuple[pd.DataFrame, List, List]:
    if False:
        while True:
            i = 10
    'Performs basic data sanity checks and ordering,\n    as well as prepare dataframe for fitting or predicting.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            containing column ``ds``\n        check_y : bool\n            if df must have series values\n            set to True if training or predicting with autoregression\n        covariates : list or dict\n            covariate column names\n        regressors : list or dict\n            regressor column names\n        events : list or dict\n            event column names\n        seasonalities : list or dict\n            seasonalities column names\n        future : bool\n            if df is a future dataframe\n\n    Returns\n    -------\n        pd.DataFrame or dict\n            checked dataframe\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    if df.groupby('ID').size().min() < 1:
        raise ValueError('Dataframe has no rows.')
    if 'ds' not in df:
        raise ValueError("Dataframe must have columns 'ds' with the dates.")
    if df['ds'].isnull().any():
        raise ValueError('Found NaN in column ds.')
    if not np.issubdtype(df['ds'].to_numpy().dtype, np.datetime64):
        df['ds'] = pd.to_datetime(df.loc[:, 'ds'], utc=True).dt.tz_convert(None)
    if df.groupby('ID').apply(lambda x: x.duplicated('ds').any()).any():
        raise ValueError('Column ds has duplicate values. Please remove duplicates.')
    regressors_to_remove = []
    lag_regressors_to_remove = []
    columns = []
    if check_y:
        columns.append('y')
    if regressors is not None:
        for reg in regressors:
            if len(df[reg].unique()) < 2:
                log.warning('Encountered future regressor with only unique values in training set across all IDs.Automatically removed variable.')
                regressors_to_remove.append(reg)
        if isinstance(regressors, list):
            columns.extend(regressors)
        else:
            columns.extend(regressors.keys())
    if covariates is not None:
        for covar in covariates:
            if len(df[covar].unique()) < 2:
                log.warning('Encountered lagged regressor with only unique values in training set across all IDs.Automatically removed variable.')
                lag_regressors_to_remove.append(covar)
        if isinstance(covariates, list):
            columns.extend(covariates)
        else:
            columns.extend(covariates.keys())
    if events is not None:
        if isinstance(events, list):
            columns.extend(events)
        else:
            columns.extend(events.keys())
    if seasonalities is not None:
        for season in seasonalities.periods:
            condition_name = seasonalities.periods[season].condition_name
            if condition_name is not None:
                if not df[condition_name].isin([True, False]).all() and (not df[condition_name].between(0, 1).all()):
                    raise ValueError(f'Condition column {condition_name} must be boolean or numeric between 0 and 1.')
                columns.append(condition_name)
    for name in columns:
        if name not in df:
            raise ValueError(f'Column {name!r} missing from dataframe')
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError(f'Dataframe column {name!r} only has NaN rows.')
        if not np.issubdtype(df[name].dtype, np.number):
            df[name] = pd.to_numeric(df[name])
        if np.isinf(df.loc[:, name].values).any():
            df.loc[:, name] = df[name].replace([np.inf, -np.inf], np.nan)
        if df.loc[df.loc[:, name].notnull()].shape[0] < 1:
            raise ValueError(f'Dataframe column {name!r} only has NaN rows.')
    if future:
        return (df, regressors_to_remove, lag_regressors_to_remove)
    if len(regressors_to_remove) > 0:
        regressors_to_remove = list(set(regressors_to_remove))
        df = df.drop(regressors_to_remove, axis=1)
        assert df is not None
    if len(lag_regressors_to_remove) > 0:
        lag_regressors_to_remove = list(set(lag_regressors_to_remove))
        df = df.drop(lag_regressors_to_remove, axis=1)
        assert df is not None
    return (df, regressors_to_remove, lag_regressors_to_remove)

def _crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
    if False:
        while True:
            i = 10
    'Splits data in k folds for crossvalidation.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        k : int\n            number of CV folds\n        fold_pct : float\n            percentage of overall samples to be in each fold\n        fold_overlap_pct : float\n            percentage of overlap between the validation folds (default: 0.0)\n\n    Returns\n    -------\n        list of k tuples [(df_train, df_val), ...]\n\n            training data\n\n            validation data\n    '
    assert len(df['ID'].unique()) == 1
    if n_lags == 0:
        assert n_forecasts == 1
    total_samples = len(df) - n_lags + 2 - 2 * n_forecasts
    samples_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    assert min_train >= samples_fold
    folds = []
    df_fold = df.copy(deep=True)
    for i in range(k, 0, -1):
        (df_train, df_val) = split_df(df_fold, n_lags, n_forecasts, valid_p=samples_fold, inputs_overbleed=True)
        folds.append((df_train, df_val))
        split_idx = len(df_fold) - samples_fold + samples_overlap
        df_fold = df_fold.iloc[:split_idx].reset_index(drop=True)
    folds = folds[::-1]
    return folds

def find_valid_time_interval_for_cv(df):
    if False:
        print('Hello World!')
    'Find time interval of interception among all the time series from dict.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data with column ``ds``, ``y``, and ``ID``\n\n    Returns\n    -------\n        str\n            time interval start\n        str\n            time interval end\n    '
    time_interval_intersection = df[df['ID'] == df['ID'].iloc[0]]['ds']
    for (df_name, df_i) in df.groupby('ID'):
        time_interval_intersection = pd.merge(time_interval_intersection, df_i, how='inner', on=['ds'])
        time_interval_intersection = time_interval_intersection[['ds']]
    start_date = time_interval_intersection['ds'].iloc[0]
    end_date = time_interval_intersection['ds'].iloc[-1]
    return (start_date, end_date)

def unfold_dict_of_folds(folds_dict, k):
    if False:
        for i in range(10):
            print('nop')
    'Convert dict of folds for typical format of folding of train and test data.\n\n    Parameters\n    ----------\n        folds_dict : dict\n            dict of folds\n        k : int\n            number of folds initially set\n\n    Returns\n    -------\n        list of k tuples [(df_train, df_val), ...]\n\n            training data\n\n            validation data\n    '
    folds = []
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for j in range(0, k):
        for key in folds_dict:
            assert k == len(folds_dict[key])
            df_train = pd.concat((df_train, folds_dict[key][j][0]), ignore_index=True)
            df_test = pd.concat((df_test, folds_dict[key][j][1]), ignore_index=True)
        folds.append((df_train, df_test))
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
    return folds

def _crossvalidation_with_time_threshold(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0):
    if False:
        return 10
    'Splits data in k folds for crossvalidation accordingly to time threshold.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data with column ``ds``, ``y``, and ``ID``\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        k : int\n            number of CV folds\n        fold_pct : float\n            percentage of overall samples to be in each fold\n        fold_overlap_pct : float\n            percentage of overlap between the validation folds (default: 0.0)\n\n    Returns\n    -------\n        list of k tuples [(df_train, df_val), ...]\n\n            training data\n\n            validation data\n    '
    df_merged = merge_dataframes(df)
    total_samples = len(df_merged) - n_lags + 2 - 2 * n_forecasts
    samples_fold = max(1, int(fold_pct * total_samples))
    samples_overlap = int(fold_overlap_pct * samples_fold)
    assert samples_overlap < samples_fold
    min_train = total_samples - samples_fold - (k - 1) * (samples_fold - samples_overlap)
    assert min_train >= samples_fold
    folds = []
    (df_fold, _, _, _) = prep_or_copy_df(df)
    for i in range(k, 0, -1):
        threshold_time_stamp = find_time_threshold(df_fold, n_lags, n_forecasts, samples_fold, inputs_overbleed=True)
        (df_train, df_val) = split_considering_timestamp(df_fold, n_lags, n_forecasts, inputs_overbleed=True, threshold_time_stamp=threshold_time_stamp)
        folds.append((df_train, df_val))
        split_idx = len(df_merged) - samples_fold + samples_overlap
        df_merged = df_merged[:split_idx].reset_index(drop=True)
        threshold_time_stamp = df_merged['ds'].iloc[-1]
        df_fold_aux = pd.DataFrame()
        for (df_name, df_i) in df_fold.groupby('ID'):
            df_aux = df_i.copy(deep=True).iloc[:len(df_i[df_i['ds'] < threshold_time_stamp]) + 1].reset_index(drop=True)
            df_fold_aux = pd.concat((df_fold_aux, df_aux), ignore_index=True)
        df_fold = df_fold_aux.copy(deep=True)
    folds = folds[::-1]
    return folds

def crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct=0.0, global_model_cv_type='global-time'):
    if False:
        print('Hello World!')
    'Splits data in k folds for crossvalidation.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        k : int\n            number of CV folds\n        fold_pct : float\n            percentage of overall samples to be in each fold\n        fold_overlap_pct : float\n            percentage of overlap between the validation folds (default: 0.0)\n        global_model_cv_type : str\n            Type of crossvalidation to apply to the time series.\n\n                options:\n\n                    ``global-time`` (default) crossvalidation is performed according to a time stamp threshold.\n\n                    ``local`` each episode will be crossvalidated locally (may cause time leakage among different\n                    episodes)\n\n                    ``intersect`` only the time intersection of all the episodes will be considered. A considerable\n                    amount of data may not be used. However, this approach guarantees an equal number of train/test\n                    samples for each episode.\n\n    Returns\n    -------\n        list of k tuples [(df_train, df_val), ...]\n\n            training data\n\n            validation data\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    folds = []
    if len(df['ID'].unique()) == 1:
        for (df_name, df_i) in df.groupby('ID'):
            folds = _crossvalidation_split_df(df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
    elif global_model_cv_type == 'global-time' or global_model_cv_type is None:
        folds = _crossvalidation_with_time_threshold(df, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
    elif global_model_cv_type == 'local':
        folds_dict = {}
        for (df_name, df_i) in df.groupby('ID'):
            folds_dict[df_name] = _crossvalidation_split_df(df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
        folds = unfold_dict_of_folds(folds_dict, k)
    elif global_model_cv_type == 'intersect':
        folds_dict = {}
        (start_date, end_date) = find_valid_time_interval_for_cv(df)
        for (df_name, df_i) in df.groupby('ID'):
            mask = (df_i['ds'] >= start_date) & (df_i['ds'] <= end_date)
            df_i = df_i[mask].copy(deep=True)
            folds_dict[df_name] = _crossvalidation_split_df(df_i, n_lags, n_forecasts, k, fold_pct, fold_overlap_pct)
        folds = unfold_dict_of_folds(folds_dict, k)
    else:
        raise ValueError('Please choose a valid type of global model crossvalidation (i.e. global-time, local, or intersect)')
    return folds

def double_crossvalidation_split_df(df, n_lags, n_forecasts, k, valid_pct, test_pct):
    if False:
        i = 10
        return i + 15
    'Splits data in two sets of k folds for crossvalidation on validation and test data.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        k : int\n            number of CV folds\n        valid_pct : float\n            percentage of overall samples to be in validation\n        test_pct : float\n            percentage of overall samples to be in test\n\n    Returns\n    -------\n        tuple of k tuples [(folds_val, folds_test), …]\n            elements same as :meth:`crossvalidation_split_df` returns\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    if len(df['ID'].unique()) > 1:
        raise NotImplementedError('double_crossvalidation_split_df not implemented for df with many time series')
    fold_pct_test = float(test_pct) / k
    folds_test = crossvalidation_split_df(df, n_lags, n_forecasts, k, fold_pct=fold_pct_test, fold_overlap_pct=0.0)
    df_train = folds_test[0][0]
    fold_pct_val = float(valid_pct) / k / (1.0 - test_pct)
    folds_val = crossvalidation_split_df(df_train, n_lags, n_forecasts, k, fold_pct=fold_pct_val, fold_overlap_pct=0.0)
    return (folds_val, folds_test)

def find_time_threshold(df, n_lags, n_forecasts, valid_p, inputs_overbleed):
    if False:
        print('Hello World!')
    'Find time threshold for dividing timeseries into train and validation sets.\n    Prevents overbleed of targets. Overbleed of inputs can be configured.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data with column ``ds``, ``y``, and ``ID``\n        n_lags : int\n            identical to NeuralProphet\n        valid_p : float\n            fraction (0,1) of data to use for holdout validation set\n        inputs_overbleed : bool\n            Whether to allow last training targets to be first validation inputs (never targets)\n\n    Returns\n    -------\n        str\n            time stamp threshold defines the boundary for the train and validation sets split.\n    '
    df_merged = merge_dataframes(df)
    n_samples = len(df_merged) - n_lags + 2 - 2 * n_forecasts
    n_samples = n_samples if inputs_overbleed else n_samples - n_lags
    if 0.0 < valid_p < 1.0:
        n_valid = max(1, int(n_samples * valid_p))
    else:
        assert valid_p >= 1
        assert isinstance(valid_p, int)
        n_valid = valid_p
    n_train = n_samples - n_valid
    threshold_time_stamp = df_merged.loc[n_train, 'ds']
    log.debug('Time threshold: ', threshold_time_stamp)
    return threshold_time_stamp

def split_considering_timestamp(df, n_lags, n_forecasts, inputs_overbleed, threshold_time_stamp):
    if False:
        print('Hello World!')
    'Splits timeseries into train and validation sets according to given threshold_time_stamp.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            data with column ``ds``, ``y``, and ``ID``\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        inputs_overbleed : bool\n            Whether to allow last training targets to be first validation inputs (never targets)\n        threshold_time_stamp : str\n            time stamp boundary that defines splitting of data\n\n    Returns\n    -------\n        pd.DataFrame, dict\n            training data\n        pd.DataFrame, dict\n            validation data\n    '
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    for (df_name, df_i) in df.groupby('ID'):
        if df[df['ID'] == df_name]['ds'].max() < threshold_time_stamp:
            df_train = pd.concat((df_train, df_i.copy(deep=True)), ignore_index=True)
        elif df[df['ID'] == df_name]['ds'].min() > threshold_time_stamp:
            df_val = pd.concat((df_val, df_i.copy(deep=True)), ignore_index=True)
        else:
            df_aux = df_i.copy(deep=True)
            n_train = len(df_aux[df_aux['ds'] < threshold_time_stamp])
            split_idx_train = n_train + n_lags + n_forecasts - 1
            split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
            df_train = pd.concat((df_train, df_aux.iloc[:split_idx_train]), ignore_index=True)
            df_val = pd.concat((df_val, df_aux.iloc[split_idx_val:]), ignore_index=True)
    return (df_train, df_val)

def split_df(df: pd.DataFrame, n_lags: int, n_forecasts: int, valid_p: float=0.2, inputs_overbleed: bool=True, local_split: bool=False):
    if False:
        while True:
            i = 10
    'Splits timeseries df into train and validation sets.\n\n    Prevents overbleed of targets. Overbleed of inputs can be configured.\n    In case of global modeling the split could be either local or global.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y``, and optionally``ID`` with all data\n        n_lags : int\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        valid_p : float, int\n            fraction (0,1) of data to use for holdout validation set, or number of validation samples >1\n        inputs_overbleed : bool\n            Whether to allow last training targets to be first validation inputs (never targets)\n        local_split : bool\n            when set to true, each episode from a dict of dataframes will be split locally\n\n    Returns\n    -------\n        pd.DataFrame, dict\n            training data\n        pd.DataFrame, dict\n            validation data\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    if local_split:
        n_samples = df.groupby('ID').size()
        n_samples = n_samples - n_lags + 2 - 2 * n_forecasts
        n_samples = n_samples if inputs_overbleed else n_samples - n_lags
        if 0.0 < valid_p < 1.0:
            n_valid = n_samples.apply(lambda x: max(1, int(x * valid_p)))
        else:
            assert valid_p >= 1
            assert isinstance(valid_p, int)
            n_valid = valid_p
        n_train = n_samples - n_valid
        log.debug(f'{n_train} n_train, {n_samples - n_train} n_eval')
    else:
        threshold_time_stamp = find_time_threshold(df, n_lags, n_forecasts, valid_p, inputs_overbleed)
        n_train = df['ds'].groupby(df['ID']).apply(lambda x: x[x < threshold_time_stamp].count())
    assert n_train.min() > 1
    split_idx_train = n_train + n_lags + n_forecasts - 1
    split_idx_val = split_idx_train - n_lags if inputs_overbleed else split_idx_train
    df_train = df.groupby('ID', group_keys=False).apply(lambda x: x.iloc[:split_idx_train[x.name]])
    df_val = df.groupby('ID', group_keys=False).apply(lambda x: x.iloc[split_idx_val[x.name]:])
    return (df_train, df_val)

def make_future_df(df_columns, last_date, periods, freq, config_events: Optional[ConfigEvents]=None, events_df=None, config_regressors=None, regressors_df=None):
    if False:
        i = 10
        return i + 15
    'Extends df periods number steps into future.\n\n    Parameters\n    ----------\n        df_columns : pd.DataFrame\n            Dataframe columns\n        last_date : pd.Datetime\n            last history date\n        periods : int\n            number of future steps to predict\n        freq : str\n            Data step sizes. Frequency of data recording, any valid frequency\n            for pd.date_range, such as ``D`` or ``M``\n        config_events : configure.ConfigEvents\n            User specified events configs\n        events_df : pd.DataFrame\n            containing column ``ds`` and ``event``\n        config_regressors : configure.ConfigFutureRegressors\n            configuration for user specified regressors,\n        regressors_df : pd.DataFrame\n            containing column ``ds`` and one column for each of the external regressors\n\n    Returns\n    -------\n        pd.DataFrame\n            input df with ``ds`` extended into future, and ``y`` set to None\n    '
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)
    future_dates = future_dates[future_dates > last_date]
    future_dates = future_dates[:periods]
    future_df = pd.DataFrame({'ds': future_dates})
    if config_events is not None:
        future_df = convert_events_to_features(future_df, config_events=config_events, events_df=events_df)
    if config_regressors is not None and regressors_df is not None:
        for regressor in regressors_df:
            future_df[regressor] = regressors_df[regressor]
    for column in df_columns:
        if column not in future_df.columns:
            if column != 't' and column != 'y_scaled':
                future_df[column] = None
    future_df.reset_index(drop=True, inplace=True)
    return future_df

def convert_events_to_features(df, config_events: ConfigEvents, events_df):
    if False:
        print('Hello World!')
    '\n    Converts events information into binary features of the df\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            Dataframe with columns ``ds`` datestamps and ``y`` time series values\n        config_events : configure.ConfigEvents\n            User specified events configs\n        events_df : pd.DataFrame\n            containing column ``ds`` and ``event``\n\n    Returns\n    -------\n        pd.DataFrame\n            input df with columns for user_specified features\n    '
    for event in config_events.keys():
        event_feature = pd.Series(0, index=range(df.shape[0]), dtype='float32')
        if events_df is None:
            dates = None
        else:
            dates = events_df[events_df.event == event].ds
            df.reset_index(drop=True, inplace=True)
            event_feature[df.ds.isin(dates)] = 1.0
        df[event] = event_feature
    return df

def add_missing_dates_nan(df, freq):
    if False:
        while True:
            i = 10
    'Fills missing datetimes in ``ds``, with NaN for all other columns except ``ID``.\n\n    Parameters\n    ----------\n        df : pd.Dataframe\n            with column ``ds``  datetimes\n        freq : str\n            Frequency of data recording, any valid frequency for pd.date_range,\n            such as ``D`` or ``M``\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe without date-gaps but nan-values\n    '
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df_resampled = df.resample(freq).asfreq()
    if 'ID' in df.columns:
        df_resampled['ID'].fillna(df['ID'].iloc[0], inplace=True)
    df_resampled.reset_index(inplace=True)
    num_added = len(df_resampled) - len(df)
    return (df_resampled, num_added)

def create_dummy_datestamps(df, freq='S', startyear=1970, startmonth=1, startday=1, starthour=0, startminute=0, startsecond=0):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to create a dummy series of datestamps for equidistant data without ds.\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe with column \'y\' and without column \'ds\'\n        freq : str\n            Frequency of data recording, any valid frequency for pd.date_range, such as ``D`` or ``M``\n        startyear, startmonth, startday, starthour, startminute, startsecond : int\n            Defines the first datestamp\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with dummy equidistant datestamps\n\n    Examples\n    --------\n    Adding dummy datestamps to a dataframe without datestamps.\n    To prepare the dataframe for training, import df_utils and insert your prefered dates.\n        >>> from neuralprophet import df_utils\n        >>> df_drop = df.drop("ds", axis=1)\n        >>> df_dummy = df_utils.create_dummy_datestamps(\n        >>> df_drop, freq="S", startyear=1970, startmonth=1, startday=1, starthour=0, startminute=0, startsecond=0\n        >>> )\n    '
    if 'ds' in df:
        raise ValueError("Column 'ds' in df detected.")
    log.info(f'Dummy equidistant datestamps added. Frequency={freq}.')
    df_length = len(df)
    startdate = pd.Timestamp(year=startyear, month=startmonth, day=startday, hour=starthour, minute=startminute, second=startsecond)
    datestamps = pd.date_range(startdate, periods=df_length, freq=freq)
    df_dummy = pd.DataFrame({'ds': datestamps, 'y': df['y']})
    return df_dummy

def fill_linear_then_rolling_avg(series, limit_linear, rolling):
    if False:
        for i in range(10):
            print('nop')
    'Adds missing dates, fills missing values with linear imputation or trend.\n\n    Parameters\n    ----------\n        series : pd.Series\n            series with nan to be filled in.\n        limit_linear : int\n            maximum number of missing values to impute.\n\n            Note\n            ----\n            because imputation is done in both directions, this value is effectively doubled.\n\n        rolling : int\n            maximal number of missing values to impute.\n\n            Note\n            ----\n            window width is rolling + 2*limit_linear\n\n    Returns\n    -------\n        pd.DataFrame\n            manipulated dataframe containing filled values\n    '
    series = pd.to_numeric(series)
    series = series.interpolate(method='linear', limit=limit_linear, limit_direction='both')
    is_na = pd.isna(series)
    rolling_avg = series.rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    series.loc[is_na] = rolling_avg[is_na]
    remaining_na = sum(series.isnull())
    return (series, remaining_na)

def get_freq_dist(ds_col):
    if False:
        return 10
    'Get frequency distribution of ``ds`` column.\n\n    Parameters\n    ----------\n        ds_col : pd.DataFrame\n            ``ds`` column of dataframe\n\n    Returns\n    -------\n        tuple\n            numeric delta values (``ms``) and distribution of frequency counts\n    '
    converted_ds = pd.to_datetime(ds_col, utc=True).view(dtype=np.int64)
    diff_ds = np.unique(converted_ds.diff(), return_counts=True)
    return diff_ds

def convert_str_to_num_freq(freq_str):
    if False:
        print('Hello World!')
    'Convert frequency tags into numeric delta in ms\n\n    Parameters\n    ----------\n        freq_str str\n            frequency tag\n\n    Returns\n    -------\n        numeric\n            frequency numeric delta in ms\n    '
    if freq_str is None:
        freq_num = 0
    else:
        aux_ts = pd.DataFrame(pd.date_range('1994-01-01', periods=100, freq=freq_str))
        (frequencies, distribution) = get_freq_dist(aux_ts[0])
        freq_num = frequencies[np.argmax(distribution)]
    return freq_num

def convert_num_to_str_freq(freq_num, initial_time_stamp):
    if False:
        for i in range(10):
            print('nop')
    'Convert numeric frequencies into frequency tags\n\n    Parameters\n    ----------\n        freq_num : int\n            numeric values of delta in ms\n        initial_time_stamp : str\n            initial time stamp of data\n\n    Returns\n    -------\n        str\n            frequency tag\n    '
    aux_ts = pd.date_range(initial_time_stamp, periods=100, freq=pd.to_timedelta(freq_num))
    freq_str = pd.infer_freq(aux_ts)
    return freq_str

def get_dist_considering_two_freqs(dist):
    if False:
        for i in range(10):
            print('nop')
    'Add occasions of the two most common frequencies\n\n    Note\n    ----\n    Useful for the frequency exceptions (i.e. ``M``, ``Y``, ``Q``, ``B``, and ``BH``).\n\n    Parameters\n    ----------\n        dist : list\n            list of occasions of frequencies\n\n    Returns\n    -------\n        numeric\n            sum of the two most common frequencies occasions\n    '
    f1 = dist.max()
    dist = np.delete(dist, np.argmax(dist))
    f2 = dist.max()
    return f1 + f2

def _get_dominant_frequency_percentage(frequencies, distribution, filter_list) -> float:
    if False:
        return 10
    'Calculate dominant frequency percentage of dataframe.\n\n    Parameters\n    ----------\n        frequencies : list\n            list of numeric delta values (``ms``) of frequencies\n        distribution : list\n            list of occasions of frequencies\n        filter_list : list\n            list of frequencies to be filtered\n\n    Returns\n    -------\n        float\n            Percentage of dominant frequency within the whole dataframe\n\n    '
    dominant_frequencies = [freq for freq in frequencies if freq in filter_list]
    dominant_distribution = [distribution[np.where(frequencies == freq)] for freq in dominant_frequencies]
    return sum(dominant_distribution) / sum(distribution)

def _infer_frequency(df, freq, min_freq_percentage=0.7):
    if False:
        i = 10
        return i + 15
    'Automatically infers frequency of dataframe.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            Dataframe with columns ``ds`` datestamps and ``y`` time series values\n        freq : str\n            Data step sizes, i.e. frequency of data recording,\n\n            Note\n            ----\n            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto``\n            (default) to automatically set frequency.\n\n        min_freq_percentage : float\n            threshold for defining major frequency of data (default: ``0.7``\n\n    Returns\n    -------\n        str\n            Valid frequency tag according to major frequency.\n\n    '
    (frequencies, distribution) = get_freq_dist(df['ds'])
    argmax_frequency = frequencies[np.argmax(distribution)]
    MONTHLY_FREQUENCIES = [2419200000000000.0, 2505600000000000.0, 2592000000000000.0, 2678400000000000.0]
    if argmax_frequency in MONTHLY_FREQUENCIES:
        dominant_freq_percentage = _get_dominant_frequency_percentage(frequencies, distribution, MONTHLY_FREQUENCIES)
        num_freq = 2678400000000000.0
        inferred_freq = 'MS' if pd.to_datetime(df['ds'].iloc[0]).day < 15 else 'M'
    elif argmax_frequency == 3.1536e+16 or argmax_frequency == 3.16224e+16:
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df['ds'])
        num_freq = 3.1536e+16
        inferred_freq = 'YS' if pd.to_datetime(df['ds'].iloc[0]).day < 15 else 'Y'
    elif argmax_frequency == 7948800000000000.0 and frequencies[np.argsort(distribution, axis=0)[-2]] == 7862400000000000.0:
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df['ds'])
        num_freq = 7948800000000000.0
        inferred_freq = 'QS' if pd.to_datetime(df['ds'].iloc[0]).day < 15 else 'Q'
    elif argmax_frequency == 86400000000000.0 and frequencies[np.argsort(distribution, axis=0)[-2]] == 259200000000000.0 and (distribution[np.argsort(distribution, axis=0)[-2]] / len(df['ds']) >= 0.12):
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df['ds'])
        num_freq = 86400000000000.0
        inferred_freq = 'B'
    elif argmax_frequency == 3600000000000.0 and frequencies[np.argsort(distribution, axis=0)[-2]] == 61200000000000.0 and (distribution[np.argsort(distribution, axis=0)[-2]] / len(df['ds']) >= 0.08):
        dominant_freq_percentage = get_dist_considering_two_freqs(distribution) / len(df['ds'])
        num_freq = 3600000000000.0
        inferred_freq = 'BH'
    else:
        dominant_freq_percentage = distribution.max() / len(df['ds'])
        num_freq = argmax_frequency
        inferred_freq = convert_num_to_str_freq(num_freq, df['ds'].iloc[0])
    log.info(f'Major frequency {inferred_freq} corresponds to {np.round(dominant_freq_percentage * 100, 3)}% of the data.')
    ideal_freq_exists = True if dominant_freq_percentage >= min_freq_percentage else False
    if ideal_freq_exists:
        if freq == 'auto' or freq is None:
            freq_str = inferred_freq
            log.info(f'Dataframe freq automatically defined as {freq_str}')
        else:
            freq_str = freq
            if convert_str_to_num_freq(freq) != convert_str_to_num_freq(inferred_freq):
                log.warning(f'Defined frequency {freq_str} is different than major frequency {inferred_freq}')
            else:
                if freq_str in ['M', 'MS', 'Q', 'QS', 'Y', 'YS']:
                    freq_str = inferred_freq
                log.info(f'Defined frequency is equal to major frequency - {freq_str}')
    elif freq == 'auto' or freq is None:
        log.warning('The auto-frequency feature is not able to detect the following frequencies: SM, BM, CBM, SMS, BMS,                     CBMS, BQ, BQS, BA, or, BAS. If the frequency of the dataframe is any of the mentioned please                         define it manually.')
        raise ValueError('Detected multiple frequencies in the timeseries please pre-process data.')
    else:
        freq_str = freq
        log.warning(f'Dataframe has multiple frequencies. It will be resampled according to given freq {freq}. Ignore                     message if actual frequency is any of the following:  SM, BM, CBM, SMS, BMS, CBMS, BQ, BQS, BA,                         or, BAS.')
    return freq_str

def infer_frequency(df, freq, n_lags, min_freq_percentage=0.7):
    if False:
        return 10
    'Automatically infers frequency of dataframe.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            Dataframe with columns ``ds`` datestamps and ``y`` time series values, and optionally``ID``\n        freq : str\n            Data step sizes, i.e. frequency of data recording,\n\n            Note\n            ----\n            Any valid frequency for pd.date_range, such as ``5min``, ``D``, ``MS`` or ``auto`` (default) to\n            automatically set frequency.\n        n_lags : int\n            identical to NeuralProphet\n        min_freq_percentage : float\n            threshold for defining major frequency of data (default: ``0.7``\n\n\n\n    Returns\n    -------\n        str\n            Valid frequency tag according to major frequency.\n\n    '
    (df, _, _, _) = prep_or_copy_df(df)
    freq_df = list()
    for (df_name, df_i) in df.groupby('ID'):
        freq_df.append(_infer_frequency(df_i, freq, min_freq_percentage))
    if len(set(freq_df)) != 1 and n_lags > 0:
        raise ValueError('One or more dataframes present different major frequencies, please make sure all dataframes present the                 same major frequency for auto-regression')
    elif len(set(freq_df)) != 1 and n_lags == 0:
        freq_str = max(set(freq_df), key=freq_df.count)
        log.warning(f'One or more major frequencies are different - setting main frequency as {freq_str}')
    else:
        freq_str = freq_df[0]
    return freq_str

def create_dict_for_events_or_regressors(df: pd.DataFrame, other_df: Optional[pd.DataFrame], other_df_name: str) -> dict:
    if False:
        return 10
    "Create a dict for events or regressors according to input df.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            Dataframe with columns ``ds`` datestamps and ``y`` time series values\n        other_df : pd.DataFrame\n            Dataframe with events or regressors\n        other_df_name : str\n            Definition of other_df (i.e. 'events', 'regressors')\n\n    Returns\n    -------\n        dict\n            dictionary with events or regressors\n    "
    df_names = list(df['ID'])
    if other_df is None:
        return {df_name: None for df_name in df_names}
    (other_df, received_ID_col, _, _) = prep_or_copy_df(other_df)
    if not received_ID_col:
        other_df = other_df.drop('ID', axis=1)
        return {df_name: other_df.copy(deep=True) for df_name in df_names}
    (df_unique_names, other_df_unique_names) = (list(df['ID'].unique()), list(other_df['ID'].unique()))
    missing_names = [name for name in other_df_unique_names if name not in df_unique_names]
    if len(missing_names) > 0:
        raise ValueError(f'ID(s) {missing_names} from {other_df_name} df is not valid - missing from original df ID column')
    df_other_dict = {}
    for df_name in df_unique_names:
        if df_name in other_df_unique_names:
            df_aux = other_df[other_df['ID'] == df_name].reset_index(drop=True).copy(deep=True)
            df_aux.drop('ID', axis=1, inplace=True)
        else:
            df_aux = None
        df_other_dict[df_name] = df_aux
    log.debug(f'Original df and {other_df_name} df are compatible')
    return df_other_dict

def handle_negative_values(df, col, handle_negatives):
    if False:
        i = 10
        return i + 15
    '\n    Handles negative values in a column according to the handle_negatives parameter.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y`` with all data\n        col : str\n            name of the regressor column\n        handle_negatives : str, int, float\n            specified handling of negative values in the regressor column. Can be one of the following options:\n\n            Options\n                    * ``remove``: Remove all negative values of the regressor.\n                    * ``error``: Raise an error in case of a negative value.\n                    * ``float`` or ``int``: Replace negative values with the provided value.\n                    * (default) ``None``: Do not handle negative values.\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with handled negative values\n    '
    if handle_negatives == 'error':
        if (df[col] < 0).any():
            raise ValueError(f'The regressor {col} contains negative values. Please preprocess data manually.')
    elif handle_negatives == 'remove':
        log.info(f"Removing {df[col].count() - (df[col] >= 0).sum()} negative value(s) from regressor {col} due to                 handle_negatives='remove'")
        df = df[df[col] >= 0]
    elif type(handle_negatives) in [int, float]:
        df.loc[df[col] < 0, col] = handle_negatives
    return df

def drop_missing_from_df(df, drop_missing, predict_steps, n_lags):
    if False:
        i = 10
        return i + 15
    'Drops windows of missing values in df according to the (lagged) samples that are dropped from TimeDataset.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y`` with all data\n        drop_missing : bool\n            identical to NeuralProphet\n        n_forecasts : int\n            identical to NeuralProphet\n        n_lags : int\n            identical to NeuralProphet\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with dropped NaN windows\n    '
    if not drop_missing:
        return df
    if n_lags == 0:
        return df
    while pd.isnull(df['y'][:-predict_steps]).any():
        window = []
        all_nan_idx = df[:-predict_steps].loc[df['y'][:-predict_steps].isnull()].index
        if len(all_nan_idx) > 0:
            for i in range(len(all_nan_idx)):
                window.append(all_nan_idx[i])
                if all_nan_idx.max() == all_nan_idx[i]:
                    break
                if all_nan_idx[i + 1] - all_nan_idx[i] > 1:
                    break
            df = df.drop(df.index[window[0]:window[-1] + 1]).reset_index().drop('index', axis=1)
            if window[0] - (n_lags - 1) >= 0:
                df = df.drop(df.index[window[0] - (n_lags - 1):window[0]]).reset_index().drop('index', axis=1)
    return df

def join_dfs_after_data_drop(predicted, df, merge=False):
    if False:
        return 10
    'Creates the intersection between df and predicted, removing any dates that have been imputed and dropped in\n    NeuralProphet.predict().\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y`` with all data\n        predicted : pd.DataFrame\n            output dataframe of NeuralProphet.predict.\n        merge : bool\n            whether to merge predicted and df into one dataframe.\n            Options\n            * (default) ``False``: Returns separate dataframes\n            * ``True``: Merges predicted and df into one dataframe\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with dates removed, that have been imputed and dropped\n    '
    df['ds'] = pd.to_datetime(df['ds'])
    predicted[predicted.columns[0]] = pd.to_datetime(predicted[predicted.columns[0]])
    df_merged = pd.DataFrame()
    df_merged = pd.concat([predicted.set_index(predicted.columns[0]), df.set_index(df.columns[0])], join='inner', axis=1)
    if not merge:
        predicted = df_merged.iloc[:, :-1]
        predicted = predicted.rename_axis('ds').reset_index()
        df = df_merged.iloc[:, -1:]
        df = df.rename_axis('ds').reset_index()
        return (predicted, df)
    else:
        return df_merged.rename_axis('ds').reset_index()

def add_quarter_condition(df: pd.DataFrame):
    if False:
        print('Hello World!')
    'Adds columns for conditional seasonalities to the df.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y`` with all data\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with added columns for conditional seasonalities\n\n            Note\n            ----\n            Quarters correspond to northern hemisphere.\n    '
    df['ds'] = pd.to_datetime(df['ds'])
    df['summer'] = df['ds'].apply(lambda x: x.month in [6, 7, 8]).astype(int)
    df['winter'] = df['ds'].apply(lambda x: x.month in [12, 1, 2]).astype(int)
    df['spring'] = df['ds'].apply(lambda x: x.month in [3, 4, 5]).astype(int)
    df['fall'] = df['ds'].apply(lambda x: x.month in [9, 10, 11]).astype(int)
    return df

def add_weekday_condition(df: pd.DataFrame):
    if False:
        print('Hello World!')
    'Adds columns for conditional seasonalities to the df.\n\n    Parameters\n    ----------\n        df : pd.DataFrame\n            dataframe containing column ``ds``, ``y`` with all data\n\n    Returns\n    -------\n        pd.DataFrame\n            dataframe with added columns for conditional seasonalities\n    '
    df['ds'] = pd.to_datetime(df['ds'])
    df['weekend'] = df['ds'].apply(lambda x: x.weekday() in [5, 6]).astype(int)
    df['weekday'] = df['ds'].apply(lambda x: x.weekday() in [0, 1, 2, 3, 4]).astype(int)
    return df

def create_mask_for_prediction_frequency(prediction_frequency, ds, forecast_lag):
    if False:
        for i in range(10):
            print('nop')
    'Creates a mask for the yhat array, to select the correct values for the prediction frequency.\n    This method is only called in _reshape_raw_predictions_to_forecst_df within NeuralProphet.predict().\n\n    Parameters\n    ----------\n        prediction_frequency : dict\n            identical to NeuralProphet\n        ds : pd.Series\n            datestamps of the predictions\n        forecast_lag : int\n            current forecast lag\n\n    Returns\n    -------\n        np.array\n            mask for the yhat array\n    '
    masks = []
    for (count, (key, value)) in enumerate(prediction_frequency.items()):
        if count > 0 and forecast_lag > 1:
            target_time = value + 1
        else:
            target_time = value + forecast_lag
        if key == 'daily-hour':
            target_time = target_time % 24
            mask = ds.dt.hour == target_time
        elif key == 'weekly-day':
            target_time = target_time % 7
            mask = ds.dt.dayofweek == target_time
        elif key == 'monthly-day':
            num_days = ds.dt.daysinmonth
            target_time = target_time % num_days
            mask = (ds.dt.day == target_time).reset_index(drop=True)
        elif key == 'yearly-month':
            target_time = target_time % 12 if target_time > 12 else target_time
            target_time = 1 if target_time == 0 else target_time
            mask = ds.dt.month == target_time
        elif key == 'hourly-minute':
            target_time = target_time % 60
            mask = ds.dt.minute == target_time
        else:
            raise ValueError(f'prediction_frequency {key} not supported')
        masks.append(mask)
    mask = np.ones((len(ds),), dtype=bool)
    for m in masks:
        mask = mask & m
    return mask