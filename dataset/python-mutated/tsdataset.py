import pandas as pd
import numpy as np
import functools
import logging
from bigdl.chronos.data.utils.feature import generate_dt_features, generate_global_features
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe
from bigdl.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe
from bigdl.chronos.data.utils.roll import roll_timeseries_dataframe
from bigdl.chronos.data.utils.time_feature import time_features, gen_time_enc_arr
from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy, scale_timeseries_numpy
from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
from bigdl.chronos.data.utils.split import split_timeseries_dataframe
from bigdl.chronos.data.utils.cycle_detection import cycle_length_est
from bigdl.chronos.data.utils.quality_inspection import quality_check_timeseries_dataframe, _abnormal_value_repair
from bigdl.chronos.data.utils.utils import _to_list, _check_type, _check_col_within, _check_col_no_na, _check_is_aligned, _check_dt_is_sorted
_DEFAULT_ID_COL_NAME = 'id'
_DEFAULT_ID_PLACEHOLDER = '0'

class TSDataset:

    def __init__(self, data, repair=False, **schema):
        if False:
            for i in range(10):
                print('nop')
        '\n        TSDataset is an abstract of time series dataset.\n        Cascade call is supported for most of the transform methods.\n        '
        self.df = data
        self.deploy_mode = schema['deploy_mode']
        if not self.deploy_mode:
            (_, self.df) = quality_check_timeseries_dataframe(df=self.df, dt_col=schema['dt_col'], id_col=schema['id_col'], repair=repair)
        self.id_col = schema['id_col']
        self.dt_col = schema['dt_col']
        self.feature_col = schema['feature_col'].copy()
        self.target_col = schema['target_col'].copy()
        self.numpy_x = None
        self.numpy_y = None
        self.lookback = None
        self.horizon = None
        self.label_len = None
        self.roll_feature = None
        self.roll_target = None
        self.roll_feature_df = None
        self.roll_additional_feature = None
        self.scaler = None
        self.scaler_index = [i for i in range(len(self.target_col))]
        self.id_sensitive = None
        self._has_generate_agg_feature = False
        if not self.deploy_mode:
            self._check_basic_invariants()
        self._id_list = list(np.unique(self.df[self.id_col]))
        self._freq_certainty = False
        self._freq = None
        self._is_pd_datetime = pd.api.types.is_datetime64_any_dtype(self.df[self.dt_col].dtypes)
        if self._is_pd_datetime:
            if len(self.df[self.dt_col]) < 2:
                self._freq = None
            else:
                self._freq = self.df[self.dt_col].iloc[1] - self.df[self.dt_col].iloc[0]

    @staticmethod
    def from_pandas(df, dt_col, target_col, id_col=None, extra_feature_col=None, with_split=False, val_ratio=0, test_ratio=0.1, repair=False, deploy_mode=False):
        if False:
            while True:
                i = 10
        '\n        Initialize tsdataset(s) from pandas dataframe.\n\n        :param df: a pandas dataframe for your raw time series data.\n        :param dt_col: a str indicates the col name of datetime\n               column in the input data frame, the dt_col must be sorted\n               from past to latest respectively for each id.\n        :param target_col: a str or list indicates the col name of target column\n               in the input data frame.\n        :param id_col: (optional) a str indicates the col name of dataframe id. If\n               it is not explicitly stated, then the data is interpreted as only\n               containing a single id.\n        :param extra_feature_col: (optional) a str or list indicates the col name\n               of extra feature columns that needs to predict the target column.\n        :param with_split: (optional) bool, states if we need to split the dataframe\n               to train, validation and test set. The value defaults to False.\n        :param val_ratio: (optional) float, validation ratio. Only effective when\n               with_split is set to True. The value defaults to 0.\n        :param test_ratio: (optional) float, test ratio. Only effective when with_split\n               is set to True. The value defaults to 0.1.\n        :param repair: a bool indicates whether automaticly repair low quality data,\n               which may call .impute()/.resample() or modify datetime column on dataframe.\n               The value defaults to False.\n        :param deploy_mode: a bool indicates whether to use deploy mode, which will be used in\n               production environment to reduce the latency of data processing. The value\n               defaults to False.\n\n        :return: a TSDataset instance when with_split is set to False,\n                 three TSDataset instances when with_split is set to True.\n\n        Create a tsdataset instance by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",\n        >>>                                   target_col="value", id_col="id",\n        >>>                                   extra_feature_col=["extra feature 1",\n        >>>                                                      "extra feature 2"])\n        '
        if not deploy_mode:
            _check_type(df, 'df', pd.DataFrame)
            tsdataset_df = df.copy(deep=True)
        else:
            tsdataset_df = df
        target_col = _to_list(target_col, name='target_col', deploy_mode=deploy_mode)
        feature_col = _to_list(extra_feature_col, name='extra_feature_col', deploy_mode=deploy_mode)
        if id_col is None:
            tsdataset_df[_DEFAULT_ID_COL_NAME] = _DEFAULT_ID_PLACEHOLDER
            id_col = _DEFAULT_ID_COL_NAME
        if with_split:
            tsdataset_dfs = split_timeseries_dataframe(df=tsdataset_df, id_col=id_col, val_ratio=val_ratio, test_ratio=test_ratio)
            return [TSDataset(data=tsdataset_dfs[i], repair=repair, id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col, deploy_mode=deploy_mode) for i in range(3)]
        return TSDataset(data=tsdataset_df, repair=repair, id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col, deploy_mode=deploy_mode)

    @staticmethod
    def from_parquet(path, dt_col, target_col, id_col=None, extra_feature_col=None, with_split=False, val_ratio=0, test_ratio=0.1, repair=False, deploy_mode=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize tsdataset(s) from path of parquet file.\n\n        :param path: A string path to parquet file. The string could be a URL.\n               Valid URL schemes include hdfs, http, ftp, s3, gs, and file. For file URLs, a host\n               is expected. A local file could be: file://localhost/path/to/table.parquet.\n               A file URL can also be a path to a directory that contains multiple partitioned\n               parquet files.\n        :param dt_col: a str indicates the col name of datetime\n               column in the input data frame.\n        :param target_col: a str or list indicates the col name of target column\n               in the input data frame.\n        :param id_col: (optional) a str indicates the col name of dataframe id. If\n               it is not explicitly stated, then the data is interpreted as only\n               containing a single id.\n        :param extra_feature_col: (optional) a str or list indicates the col name\n               of extra feature columns that needs to predict the target column.\n        :param with_split: (optional) bool, states if we need to split the dataframe\n               to train, validation and test set. The value defaults to False.\n        :param val_ratio: (optional) float, validation ratio. Only effective when\n               with_split is set to True. The value defaults to 0.\n        :param test_ratio: (optional) float, test ratio. Only effective when with_split\n               is set to True. The value defaults to 0.1.\n        :param repair: a bool indicates whether automaticly repair low quality data,\n               which may call .impute()/.resample() or modify datetime column on dataframe.\n               The value defaults to False.\n        :param deploy_mode: a bool indicates whether to use deploy mode, which will be used in\n               production environment to reduce the latency of data processing. The value\n               defaults to False.\n        :param kwargs: Any additional kwargs are passed to the pd.read_parquet\n               and pyarrow.parquet.read_table.\n\n        :return: a TSDataset instance when with_split is set to False,\n                 three TSDataset instances when with_split is set to True.\n\n        Create a tsdataset instance by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> tsdataset = TSDataset.from_parquet("hdfs://path/to/table.parquet", dt_col="datetime",\n        >>>                                   target_col="value", id_col="id",\n        >>>                                   extra_feature_col=["extra feature 1",\n        >>>                                                      "extra feature 2"])\n        '
        from bigdl.chronos.data.utils.file import parquet2pd
        columns = _to_list(dt_col, name='dt_col', deploy_mode=deploy_mode) + _to_list(target_col, name='target_col', deploy_mode=deploy_mode) + _to_list(id_col, name='id_col', deploy_mode=deploy_mode) + _to_list(extra_feature_col, name='extra_feature_col', deploy_mode=deploy_mode)
        df = parquet2pd(path, columns=columns, **kwargs)
        return TSDataset.from_pandas(df, repair=repair, dt_col=dt_col, target_col=target_col, id_col=id_col, extra_feature_col=extra_feature_col, with_split=with_split, val_ratio=val_ratio, test_ratio=test_ratio, deploy_mode=deploy_mode)

    @staticmethod
    def from_prometheus(prometheus_url, query, starttime, endtime, step, target_col=None, id_col=None, extra_feature_col=None, with_split=False, val_ratio=0, test_ratio=0.1, repair=False, deploy_mode=False, **kwargs):
        if False:
            return 10
        '\n        Initialize tsdataset(s) from Prometheus data for specified time period via url.\n\n        :param prometheus_url: a str indicates url of a Prometheus server.\n        :param query: a Prometheus expression query str or list.\n        :param starttime: start timestamp of the specified time period, RFC-3339 string\n               or as a Unix timestamp in seconds.\n        :param endtime: end timestamp of the specified time period, RFC-3339 string\n               or as a Unix timestamp in seconds.\n        :param step: a str indicates query resolution step width in Prometheus duration format\n               or float number of seconds. More information about Prometheus time durations\n               are here:\n               https://prometheus.io/docs/prometheus/latest/querying/basics/#time-durations\n        :param target_col: (optional) a Prometheus expression query str or list indicates the\n               col name of target column in the input data frame. If it is not explicitly stated,\n               then target column is automatically specified according to the Prometheus data.\n        :param id_col: (optional) a Prometheus expression query str indicates the col name of\n               dataframe id. If it is not explicitly stated, then the data is interpreted as\n               only containing a single id.\n        :param extra_feature_col: (optional) a Prometheus expression query str or list indicates\n               the col name of extra feature columns that needs to predict the target column.\n               If it is not explicitly stated, then extra feature column is None.\n        :param with_split: (optional) bool, states if we need to split the dataframe\n               to train, validation and test set. The value defaults to False.\n        :param val_ratio: (optional) float, validation ratio. Only effective when\n               with_split is set to True. The value defaults to 0.\n        :param test_ratio: (optional) float, test ratio. Only effective when with_split\n               is set to True. The value defaults to 0.1.\n        :param repair: a bool indicates whether automaticly repair low quality data,\n               which may call .impute()/.resample() or modify datetime column on dataframe.\n               The value defaults to False.\n        :param deploy_mode: a bool indicates whether to use deploy mode, which will be used in\n               production environment to reduce the latency of data processing. The value\n               defaults to False.\n        :param kwargs: Any additional kwargs are passed to the Prometheus query, such as\n               timeout.\n\n        :return: a TSDataset instance when with_split is set to False,\n                 three TSDataset instances when with_split is set to True.\n\n        Create a tsdataset instance by:\n\n        >>> # Here is an example:\n        >>> tsdataset = TSDataset.from_prometheus(prometheus_url="http://localhost:9090",\n        >>>                                       query="collectd_cpufreq{cpufreq="0"}",\n        >>>                                       starttime="2022-09-01T00:00:00Z",\n        >>>                                       endtime="2022-10-01T00:00:00Z",\n        >>>                                       step="1h")\n        '
        from bigdl.chronos.data.utils.prometheus_df import GetRangeDataframe
        query_list = _to_list(query, name='query', deploy_mode=deploy_mode)
        columns = {'target_col': _to_list(target_col, name='target_col', deploy_mode=deploy_mode), 'id_col': _to_list(id_col, name='id_col', deploy_mode=deploy_mode), 'extra_feature_col': _to_list(extra_feature_col, name='extra_feature_col', deploy_mode=deploy_mode)}
        (df, df_columns) = GetRangeDataframe(prometheus_url, query_list, starttime, endtime, step, columns=columns, **kwargs)
        return TSDataset.from_pandas(df, dt_col=df_columns['dt_col'], target_col=df_columns['target_col'], id_col=df_columns['id_col'], extra_feature_col=df_columns['extra_feature_col'], with_split=with_split, val_ratio=val_ratio, test_ratio=test_ratio, repair=repair, deploy_mode=deploy_mode)

    def impute(self, mode='last', const_num=0):
        if False:
            i = 10
            return i + 15
        '\n        Impute the tsdataset by imputing each univariate time series\n        distinguished by id_col and feature_col.\n\n        :param mode: imputation mode, select from "last", "const" or "linear".\n\n            "last": impute by propagating the last non N/A number to its following N/A.\n            if there is no non N/A number ahead, 0 is filled instead.\n\n            "const": impute by a const value input by user.\n\n            "linear": impute by linear interpolation.\n        :param const_num:  indicates the const number to fill, which is only effective when mode\n            is set to "const".\n\n        :return: the tsdataset instance.\n        '
        result = []
        groups = self.df.groupby([self.id_col])
        for (_, group) in groups:
            result.append(impute_timeseries_dataframe(df=group, dt_col=self.dt_col, mode=mode, const_num=const_num))
        self.df = pd.concat(result, axis=0)
        self.df.reset_index(drop=True, inplace=True)
        return self

    def deduplicate(self):
        if False:
            return 10
        '\n        Remove those duplicated records which has exactly the same values in each feature_col\n        for each multivariate timeseries distinguished by id_col.\n\n        :return: the tsdataset instance.\n        '
        self.df = deduplicate_timeseries_dataframe(df=self.df, dt_col=self.dt_col)
        return self

    def resample(self, interval, start_time=None, end_time=None, merge_mode='mean'):
        if False:
            while True:
                i = 10
        '\n        Resample on a new interval for each univariate time series distinguished\n        by id_col and feature_col.\n\n        :param interval: pandas offset aliases, indicating time interval of the output dataframe.\n        :param start_time: start time of the output dataframe.\n        :param end_time: end time of the output dataframe.\n        :param merge_mode: if current interval is smaller than output interval,\n            we need to merge the values in a mode. "max", "min", "mean"\n            or "sum" are supported for now.\n\n        :return: the tsdataset instance.\n        '
        if not self.deploy_mode:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(self._is_pd_datetime, 'The time series data does not have a Pandas datetime format (you can use pandas.to_datetime to convert a string into a datetime format).')
            from pandas.api.types import is_numeric_dtype
            type_error_list = [val for val in self.target_col + self.feature_col if not is_numeric_dtype(self.df[val])]
            try:
                for val in type_error_list:
                    self.df[val] = self.df[val].astype(np.float32)
            except Exception:
                invalidInputError(False, 'All the columns of target_col and extra_feature_col should be of numeric type.')
        self.df = self.df.groupby([self.id_col]).apply(lambda df: resample_timeseries_dataframe(df=df, dt_col=self.dt_col, interval=interval, start_time=start_time, end_time=end_time, id_col=self.id_col, merge_mode=merge_mode, deploy_mode=self.deploy_mode))
        self._freq = pd.Timedelta(interval)
        self._freq_certainty = True
        self.df.reset_index(drop=True, inplace=True)
        return self

    def repair_abnormal_data(self, mode='relative', threshold=3.0):
        if False:
            print('Hello World!')
        '\n        Repair the tsdataset by replacing abnormal data detected based on threshold\n        with the last non N/A number.\n\n        :param mode: detect abnormal data mode, select from "absolute" or "relative".\n\n            "absolute": detect abnormal data by comparing with max and min value.\n\n            "relative": detect abnormal data by comparing with mean value plus/minus several\n            times standard deviation.\n        :param threshold: indicates the range of comparison. It is a 2-dim tuple of float\n               (min_value, max_value) when mode is set to "absolute" while it is a float\n               number when mode is set to "relative".\n\n        :return: the tsdataset instance.\n        '
        self.df = _abnormal_value_repair(df=self.df, dt_col=self.dt_col, mode=mode, threshold=threshold)
        return self

    def gen_dt_feature(self, features='auto', one_hot_features=None):
        if False:
            return 10
        '\n        Generate datetime feature(s) for each record.\n\n        :param features: str or list, states which feature(s) will be generated. If the value\n               is set to be a str, it should be one of "auto" or "all". For "auto", a subset\n               of datetime features will be generated under the consideration of the sampling\n               frequency of your data. For "all", the whole set of datetime features will be\n               generated. If the value is set to be a list, the list should contain the features\n               you want to generate. A table of all datatime features and their description is\n               listed below. The value defaults to "auto".\n        :param one_hot_features: list, states which feature(s) will be generated as one-hot-encoded\n               feature. The value defaults to None, which means no features will be generated with               one-hot-encoded.\n\n        | "MINUTE": The minute of the time stamp.\n        | "DAY": The day of the time stamp.\n        | "DAYOFYEAR": The ordinal day of the year of the time stamp.\n        | "HOUR": The hour of the time stamp.\n        | "WEEKDAY": The day of the week of the time stamp, Monday=0, Sunday=6.\n        | "WEEKOFYEAR": The ordinal week of the year of the time stamp.\n        | "MONTH": The month of the time stamp.\n        | "YEAR": The year of the time stamp.\n        | "IS_AWAKE": Bool value indicating whether it belongs to awake hours for the time stamp,\n        | True for hours between 6A.M. and 1A.M.\n        | "IS_BUSY_HOURS": Bool value indicating whether it belongs to busy hours for the time\n        | stamp, True for hours between 7A.M. and 10A.M. and hours between 4P.M. and 8P.M.\n        | "IS_WEEKEND": Bool value indicating whether it belongs to weekends for the time stamp,\n        | True for Saturdays and Sundays.\n\n        :return: the tsdataset instance.\n        '
        if not self.deploy_mode:
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(self._is_pd_datetime, 'The time series data does not have a Pandas datetime format(you can use pandas.to_datetime to convert a string into a datetime format.)')
        features_generated = []
        self.df = generate_dt_features(input_df=self.df, dt_col=self.dt_col, features=features, one_hot_features=one_hot_features, freq=self._freq, features_generated=features_generated)
        self.feature_col += features_generated
        return self

    def gen_global_feature(self, settings='comprehensive', full_settings=None, n_jobs=1):
        if False:
            return 10
        '\n        Generate per-time-series feature for each time series.\n        This method will be implemented by tsfresh.\n        Make sure that the specified column name does not contain \'__\'.\n\n        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"\n               "minimal" and "efficient". If a dict is set, then it should follow the instruction\n               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".\n        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in\n               tsfresh. The value is defaulted to None.\n        :param n_jobs: int. The number of processes to use for parallelization.\n\n        :return: the tsdataset instance.\n        '
        from bigdl.nano.utils.common import invalidInputError
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
        except ImportError:
            invalidInputError(False, 'Please install tsfresh by `pip install tsfresh` to use `gen_global_feature` method.')
        DEFAULT_PARAMS = {'comprehensive': ComprehensiveFCParameters(), 'minimal': MinimalFCParameters(), 'efficient': EfficientFCParameters()}
        invalidInputError(not self._has_generate_agg_feature, 'Only one of gen_global_feature and gen_rolling_feature should be called.')
        if full_settings is not None:
            (self.df, addtional_feature) = generate_global_features(input_df=self.df, column_id=self.id_col, column_sort=self.dt_col, kind_to_fc_parameters=full_settings, n_jobs=n_jobs)
            self.feature_col += addtional_feature
            return self
        if isinstance(settings, str):
            invalidInputError(settings in ['comprehensive', 'minimal', 'efficient'], "settings str should be one of 'comprehensive', 'minimal', 'efficient', but found {settings}.")
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings
        (self.df, addtional_feature) = generate_global_features(input_df=self.df, column_id=self.id_col, column_sort=self.dt_col, default_fc_parameters=default_fc_parameters, n_jobs=n_jobs)
        self.feature_col += addtional_feature
        self._has_generate_agg_feature = True
        return self

    def gen_rolling_feature(self, window_size, settings='comprehensive', full_settings=None, n_jobs=1):
        if False:
            i = 10
            return i + 15
        '\n        Generate aggregation feature for each sample.\n        This method will be implemented by tsfresh.\n        Make sure that the specified column name does not contain \'__\'.\n\n        :param window_size: int, generate feature according to the rolling result.\n        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"\n               "minimal" and "efficient". If a dict is set, then it should follow the instruction\n               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".\n        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in\n               tsfresh. The value is defaulted to None.\n        :param n_jobs: int. The number of processes to use for parallelization.\n\n        :return: the tsdataset instance.\n        '
        from bigdl.nano.utils.common import invalidInputError
        try:
            from tsfresh.utilities.dataframe_functions import roll_time_series
            from tsfresh.utilities.dataframe_functions import impute as impute_tsfresh
            from tsfresh import extract_features
            from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
        except ImportError:
            invalidInputError(False, 'Please install tsfresh by `pip install tsfresh` to use `gen_rolling_feature` method.')
        DEFAULT_PARAMS = {'comprehensive': ComprehensiveFCParameters(), 'minimal': MinimalFCParameters(), 'efficient': EfficientFCParameters()}
        invalidInputError(not self._has_generate_agg_feature, 'Only one of gen_global_feature and gen_rolling_feature should be called.')
        if isinstance(settings, str):
            invalidInputError(settings in ['comprehensive', 'minimal', 'efficient'], "settings str should be one of 'comprehensive', 'minimal', 'efficient', but found {settings}.")
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings
        invalidInputError(window_size < self.df.groupby(self.id_col).size().min() + 1, 'gen_rolling_feature should have a window_size smaller than shortest time series length.')
        df_rolled = roll_time_series(self.df, column_id=self.id_col, column_sort=self.dt_col, max_timeshift=window_size - 1, min_timeshift=window_size - 1, n_jobs=n_jobs)
        if not full_settings:
            self.roll_feature_df = extract_features(df_rolled, column_id=self.id_col, column_sort=self.dt_col, default_fc_parameters=default_fc_parameters, n_jobs=n_jobs)
        else:
            self.roll_feature_df = extract_features(df_rolled, column_id=self.id_col, column_sort=self.dt_col, kind_to_fc_parameters=full_settings, n_jobs=n_jobs)
        impute_tsfresh(self.roll_feature_df)
        self.feature_col += list(self.roll_feature_df.columns)
        self.roll_additional_feature = list(self.roll_feature_df.columns)
        self._has_generate_agg_feature = True
        return self

    def roll(self, horizon, lookback='auto', feature_col=None, target_col=None, id_sensitive=False, time_enc=False, label_len=0, is_predict=False):
        if False:
            while True:
                i = 10
        '\n        Sampling by rolling for machine learning/deep learning models.\n\n        :param lookback: int, lookback value. Default to \'auto\',\n               if \'auto\', the mode of time series\' cycle length will be taken as the lookback.\n        :param horizon: int or list.\n               If `horizon` is an int, we will sample `horizon` step\n               continuously after the forecasting point.\n               If `horizon` is a list, we will sample discretely according\n               to the input list. 1 means the timestamp just after the observed data.\n               specially, when `horizon` is set to 0, ground truth will be generated as None.\n        :param feature_col: str or list, indicates the feature col name. Default to None,\n               where we will take all available feature in rolling.\n        :param target_col: str or list, indicates the target col name. Default to None,\n               where we will take all target in rolling. It should be a subset of target_col\n               you used to initialize the tsdataset.\n        :param id_sensitive: bool.\n               If `id_sensitive` is False, we will rolling on each id\'s sub dataframe\n               and fuse the sampings.\n               The shape of rolling will be\n               x: (num_sample, lookback, num_feature_col + num_target_col)\n               y: (num_sample, horizon + label_len, num_target_col)\n               where num_sample is the summation of sample number of each dataframe.\n\n               If `id_sensitive` is True, we will rolling on the wide dataframe whose\n               columns are cartesian product of id_col and feature_col.\n               The shape of rolling will be\n               x: (num_sample, lookback, new_num_feature_col + new_num_target_col)\n               y: (num_sample, horizon + label_len, new_num_target_col)\n               where num_sample is the sample number of the wide dataframe,\n               new_num_feature_col is the product of the number of id and the number of feature_col,\n               new_num_target_col is the product of the number of id and the number of target_col.\n        :param time_enc: bool.\n               This parameter should be set to True only when you are using Autoformer model. With\n               time_enc to be true, 2 additional numpy ndarray will be returned when you call\n               `.to_numpy()`. Be sure to have a time type for dt_col if you set time_enc to True.\n        :param label_len: int.\n               This parameter should be set to True only when you are using Autoformer model. This\n               indicates the length of overlap area of output(y) and input(x) on time axis.\n        :param is_predict: bool.\n               This parameter indicates if the dataset will be sampled as a prediction dataset\n               (without groud truth).\n\n        :return: the tsdataset instance.\n\n        roll() can be called by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",\n        >>>                                   target_col="value", id_col="id",\n        >>>                                   extra_feature_col=["extra feature 1",\n        >>>                                                      "extra feature 2"])\n        >>> horizon, lookback = 1, 1\n        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=False)\n        >>> x, y = tsdataset.to_numpy()\n        >>> print(x, y) # x = [[[1.9, 1, 2 ]], [[2.3, 0, 9 ]]] y = [[[ 2.4 ]], [[ 2.6 ]]]\n        >>> print(x.shape, y.shape) # x.shape = (2, 1, 3) y.shape = (2, 1, 1)\n        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=True)\n        >>> x, y = tsdataset.to_numpy()\n        >>> print(x, y) # x = [[[ 1.9, 2.3, 1, 2, 0, 9 ]]] y = [[[ 2.4, 2.6]]]\n        >>> print(x.shape, y.shape) # x.shape = (1, 1, 6) y.shape = (1, 1, 2)\n\n        '
        if not self.deploy_mode:
            from bigdl.nano.utils.common import invalidInputError
            if id_sensitive and (not _check_is_aligned(self.df, self.id_col, self.dt_col)):
                invalidInputError(False, 'The time series data should be aligned if id_sensitive is set to True.')
        else:
            is_predict = True
        feature_col = _to_list(feature_col, 'feature_col', deploy_mode=self.deploy_mode) if feature_col is not None else self.feature_col
        target_col = _to_list(target_col, 'target_col', deploy_mode=self.deploy_mode) if target_col is not None else self.target_col
        if self.roll_additional_feature:
            additional_feature_col = list(set(feature_col).intersection(set(self.roll_additional_feature)))
            feature_col = list(set(feature_col) - set(self.roll_additional_feature))
            self.roll_feature = feature_col + additional_feature_col
        else:
            additional_feature_col = None
            self.roll_feature = feature_col
        self.roll_target = target_col
        num_id = len(self._id_list)
        num_feature_col = len(self.roll_feature)
        num_target_col = len(self.roll_target)
        self.id_sensitive = id_sensitive
        roll_feature_df = None if self.roll_feature_df is None else self.roll_feature_df[additional_feature_col]
        if time_enc and label_len == 0:
            label_len = max(lookback // 2, 1)
        (self.lookback, self.horizon, self.label_len) = (lookback, horizon, label_len)
        horizon_time = self.horizon
        if is_predict:
            self.horizon = 0
        if self.lookback == 'auto':
            self.lookback = self.get_cycle_length('mode', top_k=3)
        groups = self.df.groupby([self.id_col])
        rolling_result = []
        for (_, group) in groups:
            rolling_result.append(roll_timeseries_dataframe(df=group, roll_feature_df=roll_feature_df, lookback=self.lookback, horizon=self.horizon, feature_col=feature_col, target_col=target_col, label_len=label_len, deploy_mode=self.deploy_mode))
        concat_axis = 2 if id_sensitive else 0
        self.numpy_x = np.concatenate([rolling_result[i][0] for i in range(len(self._id_list))], axis=concat_axis).astype(np.float32)
        if horizon != 0 and is_predict is False or time_enc:
            self.numpy_y = np.concatenate([rolling_result[i][1] for i in range(len(self._id_list))], axis=concat_axis).astype(np.float32)
        else:
            self.numpy_y = None
        if time_enc:
            time_enc_arr = []
            for (_, group) in groups:
                time_enc_arr.append(gen_time_enc_arr(df=group, dt_col=self.dt_col, freq=self._freq, horizon_time=horizon_time, is_predict=is_predict, lookback=lookback, label_len=label_len))
            self.numpy_x_timeenc = np.concatenate([time_enc_arr[i][0] for i in range(len(self._id_list))], axis=0).astype(np.float32)
            self.numpy_y_timeenc = np.concatenate([time_enc_arr[i][1] for i in range(len(self._id_list))], axis=0).astype(np.float32)
        else:
            self.numpy_x_timeenc = None
            self.numpy_y_timeenc = None
        if self.id_sensitive:
            feature_start_idx = num_target_col * num_id
            reindex_list = [list(range(i * num_target_col, (i + 1) * num_target_col)) + list(range(feature_start_idx + i * num_feature_col, feature_start_idx + (i + 1) * num_feature_col)) for i in range(num_id)]
            reindex_list = functools.reduce(lambda a, b: a + b, reindex_list)
            sorted_index = sorted(range(len(reindex_list)), key=reindex_list.__getitem__)
            self.numpy_x = self.numpy_x[:, :, sorted_index]
        num_roll_target = len(self.roll_target)
        repeat_factor = len(self._id_list) if self.id_sensitive else 1
        scaler_index = [self.target_col.index(self.roll_target[i]) for i in range(num_roll_target)] * repeat_factor
        self.scaler_index = scaler_index
        return self

    def to_torch_data_loader(self, batch_size=32, roll=True, lookback='auto', horizon=None, feature_col=None, target_col=None, shuffle=True, time_enc=False, label_len=0, is_predict=False):
        if False:
            i = 10
            return i + 15
        '\n        Convert TSDataset to a PyTorch DataLoader with or without rolling. We recommend to use\n        to_torch_data_loader(default roll=True) if you don\'t need to output the rolled numpy array.\n        It is much more efficient than rolling separately, especially when the dataframe or lookback\n        is large.\n\n        :param batch_size: int, the batch_size for a Pytorch DataLoader. It defaults to 32.\n        :param roll: Boolean. Whether to roll the dataframe before converting to DataLoader.\n               If True, you must also specify lookback and horizon for rolling. If False, you must\n               have called tsdataset.roll() before calling to_torch_data_loader(). Default to True.\n        :param lookback: int, lookback value. Default to \'auto\',\n               the mode of time series\' cycle length will be taken as the lookback.\n        :param horizon: int or list,\n               if `horizon` is an int, we will sample `horizon` step\n               continuously after the forecasting point.\n               if `horizon` is a list, we will sample discretely according\n               to the input list.\n               specially, when `horizon` is set to 0, ground truth will be generated as None.\n        :param feature_col: str or list, indicates the feature col name. Default to None,\n               where we will take all available feature in rolling.\n        :param target_col: str or list, indicates the target col name. Default to None,\n               where we will take all target in rolling. it should be a subset of target_col\n               you used to initialize the tsdataset.\n        :param shuffle: if the dataloader is shuffled. default to True.\n        :param time_enc: bool,\n               This parameter should be set to True only when you are using Autoformer model. With\n               time_enc to be true, 2 additional numpy ndarray will be returned when you call\n               `.to_numpy()`. Be sure to have a time type for dt_col if you set time_enc to True.\n        :param label_len: int,\n               This parameter should be set to True only when you are using Autoformer model. This\n               indicates the length of overlap area of output(y) and input(x) on time axis.\n        :param is_predict: bool,\n               This parameter should be set to True only when you are processing test data without\n               accuracy evaluation. This indicates if the dataset will be sampled as a prediction\n               dataset(without groud truth).\n\n        :return: A pytorch DataLoader instance. The data returned from dataloader is in the\n                 following form:\n                1. a 3d numpy ndarray when is_predict=True or horizon=0\n                and time_enc=False\n                2. a 2-dim tuple of 3d numpy ndarray (x, y) when is_predict=False\n                and horizon != 0 and time_enc=False\n                3. a 4-dim tuple of 3d numpy ndarray (x, y, x_enc, y_enc) when\n                time_enc=True\n\n        to_torch_data_loader() can be called by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",\n        >>>                                   target_col="value", id_col="id",\n        >>>                                   extra_feature_col=["extra feature 1",\n        >>>                                                      "extra feature 2"])\n        >>> horizon, lookback = 1, 1\n        >>> data_loader = tsdataset.to_torch_data_loader(batch_size=32,\n        >>>                                              lookback=lookback,\n        >>>                                              horizon=horizon)\n        >>> # or roll outside. That might be less efficient than the way above.\n        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=False)\n        >>> x, y = tsdataset.to_numpy()\n        >>> print(x, y) # x = [[[1.9, 1, 2 ]], [[2.3, 0, 9 ]]] y = [[[ 2.4 ]], [[ 2.6 ]]]\n        >>> data_loader = tsdataset.to_torch_data_loader(batch_size=32, roll=False)\n\n        '
        from torch.utils.data import TensorDataset, DataLoader
        import torch
        from bigdl.nano.utils.common import invalidInputError
        if roll:
            if horizon is None:
                invalidInputError(False, 'You must input horizon if roll is True (default roll=True)!')
            from bigdl.chronos.data.utils.roll_dataset import RollDataset
            feature_col = _to_list(feature_col, 'feature_col') if feature_col is not None else self.feature_col
            target_col = _to_list(target_col, 'target_col') if target_col is not None else self.target_col
            if time_enc and label_len == 0:
                label_len = max(lookback // 2, 1)
            self.scaler_index = [self.target_col.index(t) for t in target_col]
            (self.lookback, self.horizon, self.label_len) = (lookback, horizon, label_len)
            if self.lookback == 'auto':
                self.lookback = self.get_cycle_length('mode', top_k=3)
            invalidInputError(not self._has_generate_agg_feature, "Currently to_torch_data_loader does not support 'gen_global_feature' and 'gen_rolling_feature' methods.")
            if isinstance(self.horizon, int):
                need_dflen = self.lookback + self.horizon
            else:
                need_dflen = self.lookback + max(self.horizon)
            if len(self.df) < need_dflen:
                invalidInputError(False, f'The length of the dataset must be larger than the sum of lookback and horizon, while get lookback+horizon={need_dflen} and the length of dataset is {len(self.df)}.')
            torch_dataset = RollDataset(self.df, dt_col=self.dt_col, freq=self._freq, lookback=self.lookback, horizon=self.horizon, feature_col=feature_col, target_col=target_col, id_col=self.id_col, time_enc=time_enc, label_len=label_len, is_predict=is_predict)
            self.roll_target = target_col
            self.roll_feature = feature_col
            batch_size = 32 if batch_size is None else batch_size
            return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            if self.numpy_x is None:
                invalidInputError(False, "Please call 'roll' method before transforming a TSDataset to torch DataLoader if roll is False!")
            if self.numpy_y is None:
                x = self.numpy_x
                return DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=shuffle)
            elif self.numpy_x_timeenc is None:
                (x, y) = self.to_numpy()
                return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float()), batch_size=batch_size, shuffle=shuffle)
            else:
                (x, y, x_enc, y_enc) = self.to_numpy()
                return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(x_enc).float(), torch.from_numpy(y_enc).float()), batch_size=batch_size, shuffle=shuffle)

    def to_tf_dataset(self, batch_size=32, shuffle=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Export a Dataset whose elements are slices of the given tensors.\n\n        :param batch_size: Number of samples per batch of computation.\n               If unspecified, batch_size will default to 32.\n\n        :return: a tf.data dataset, including x and y.\n        '
        import tensorflow as tf
        from bigdl.nano.utils.common import invalidInputError
        if self.numpy_x is None:
            invalidInputError(False, "Please call 'roll' method before transform a TSDataset to tf dataset!")
        data = tf.data.Dataset.from_tensor_slices((self.numpy_x, self.numpy_y))
        batch_size = 32 if batch_size is None else batch_size
        if shuffle:
            data = data.cache().shuffle(self.numpy_x.shape[0]).batch(batch_size)
        else:
            data = data.batch(batch_size).cache()
        return data.prefetch(tf.data.AUTOTUNE)

    def to_numpy(self):
        if False:
            return 10
        '\n        Export rolling result in form of :\n            1. a 3d numpy ndarray when is_predict=True or horizon=0\n               and time_enc=False\n            2. a 2-dim tuple of 3d numpy ndarray (x, y) when is_predict=False\n               and horizon != 0 and time_enc=False\n            3. a 4-dim tuple of 3d numpy ndarray (x, y, x_enc, y_enc) when\n               time_enc=True\n\n        :return: a 3d numpy ndarray when is_predict=True or horizon=0\n                 and time_enc=False.\n                 or a 2-dim tuple of 3d numpy ndarray (x, y) when is_predict=False\n                 and horizon != 0 and time_enc=False\n                 or a 4-dim tuple of 3d numpy ndarray (x, y, x_enc, y_enc)\n                 when time_enc=True.\n                 The ndarray is casted to float32.\n        '
        if not self.deploy_mode:
            from bigdl.nano.utils.common import invalidInputError
            if self.numpy_x is None:
                invalidInputError(False, "Please call 'roll' method before transform a TSDataset to numpy ndarray!")
        if self.numpy_y is None and self.numpy_x_timeenc is None:
            return self.numpy_x
        elif self.numpy_x_timeenc is None:
            return (self.numpy_x, self.numpy_y)
        else:
            return (self.numpy_x, self.numpy_y, self.numpy_x_timeenc, self.numpy_y_timeenc)

    def to_pandas(self):
        if False:
            i = 10
            return i + 15
        '\n        Export the pandas dataframe.\n\n        :return: the internal dataframe.\n        '
        return self.df.copy()

    def scale(self, scaler, fit=True):
        if False:
            while True:
                i = 10
        "\n        Scale the time series dataset's feature column and target column.\n\n        :param scaler: sklearn scaler instance, StandardScaler, MaxAbsScaler,\n               MinMaxScaler and RobustScaler are supported.\n        :param fit: if we need to fit the scaler. Typically, the value should\n               be set to True for training set, while False for validation and\n               test set. The value is defaulted to True.\n\n        :return: the tsdataset instance.\n\n        Assume there is a training set tsdata and a test set tsdata_test.\n        scale() should be called first on training set with default value fit=True,\n        then be called on test set with the same scaler and fit=False.\n\n        >>> from sklearn.preprocessing import StandardScaler\n        >>> scaler = StandardScaler()\n        >>> tsdata.scale(scaler, fit=True)\n        >>> tsdata_test.scale(scaler, fit=False)\n        "
        feature_col = self.feature_col
        if self.roll_additional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_additional_feature:
                    feature_col.append(feature)
        if fit and (not self.deploy_mode):
            self.df[self.target_col + feature_col] = scaler.fit_transform(self.df[self.target_col + feature_col])
        else:
            if not self.deploy_mode:
                from sklearn.utils.validation import check_is_fitted
                from bigdl.nano.utils.common import invalidInputError
                try:
                    invalidInputError(not check_is_fitted(scaler), 'scaler is not fittedd')
                except Exception:
                    invalidInputError(False, 'When calling scale for the first time, you need to set fit=True.')
            self.df[self.target_col + feature_col] = scale_timeseries_numpy(self.df[self.target_col + feature_col].values, scaler)
        self.scaler = scaler
        return self

    def unscale(self):
        if False:
            i = 10
            return i + 15
        "\n        Unscale the time series dataset's feature column and target column.\n\n        :return: the tsdataset instance.\n        "
        feature_col = self.feature_col
        if self.roll_additional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_additional_feature:
                    feature_col.append(feature)
        self.df[self.target_col + feature_col] = self.scaler.inverse_transform(self.df[self.target_col + feature_col])
        return self

    def unscale_numpy(self, data):
        if False:
            return 10
        "\n        Unscale the time series forecaster's numpy prediction result/ground truth.\n\n        :param data: a numpy ndarray with 3 dim whose shape should be exactly the\n               same with self.numpy_y.\n\n        :return: the unscaled numpy ndarray.\n        "
        return unscale_timeseries_numpy(data, self.scaler, self.scaler_index)

    def _check_basic_invariants(self, strict_check=False):
        if False:
            while True:
                i = 10
        '\n        This function contains a bunch of assertions to make sure strict rules(the invariants)\n        for the internal dataframe(self.df) must stands. If not, clear and user-friendly error\n        or warning message should be provided to the users.\n        This function will be called after each method(e.g. impute, deduplicate ...).\n        '
        _check_type(self.df, 'df', pd.DataFrame)
        _check_type(self.id_col, 'id_col', str)
        _check_type(self.dt_col, 'dt_col', str)
        _check_type(self.target_col, 'target_col', list)
        _check_type(self.feature_col, 'feature_col', list)
        _check_col_within(self.df, self.id_col)
        _check_col_within(self.df, self.dt_col)
        for target_col_name in self.target_col:
            _check_col_within(self.df, target_col_name)
        for feature_col_name in self.feature_col:
            if self.roll_additional_feature and feature_col_name in self.roll_additional_feature:
                continue
            _check_col_within(self.df, feature_col_name)
        _check_col_no_na(self.df, self.dt_col)
        _check_col_no_na(self.df, self.id_col)
        if strict_check:
            _check_dt_is_sorted(self.df, self.dt_col)

    def get_cycle_length(self, aggregate='mode', top_k=3):
        if False:
            i = 10
            return i + 15
        "\n        Calculate the cycle length of the time series in this TSDataset.\n\n        Args:\n            top_k (int): The freq with top top_k power after fft will be\n                used to check the autocorrelation. Higher top_k might be time-consuming.\n                The value is default to 3.\n            aggregate (str): Select the mode of calculation time period,\n                We only support 'min', 'max', 'mode', 'median', 'mean'.\n\n        Returns:\n            Describe the value of the time period distribution.\n        "
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(isinstance(top_k, int), f'top_k type must be int, but found {type(top_k)}.')
        invalidInputError(isinstance(aggregate, str), f'aggregate type must be str, but found {type(aggregate)}.')
        invalidInputError(aggregate.lower().strip() in ['min', 'max', 'mode', 'median', 'mean'], f"We Only support 'min' 'max' 'mode' 'median' 'mean', but found {aggregate}.")
        if len(self.target_col) == 1:
            res = []
            groups = self.df.groupby(self.id_col)
            for (_, group) in groups:
                res.append(cycle_length_est(group[self.target_col[0]].values, top_k))
            res = pd.Series(res)
        else:
            res = []
            groups = self.df.groupby(self.id_col)
            for (_, group) in groups:
                res.append(pd.DataFrame({'cycle_length': [cycle_length_est(group[col].values, top_k) for col in self.target_col]}))
            res = pd.concat(res, axis=0)
            res = res.cycle_length
        if aggregate.lower().strip() == 'mode':
            self.best_cycle_length = int(res.value_counts().index[0])
        elif aggregate.lower().strip() == 'mean':
            self.best_cycle_length = int(res.mean())
        elif aggregate.lower().strip() == 'median':
            self.best_cycle_length = int(res.median())
        elif aggregate.lower().strip() == 'min':
            self.best_cycle_length = int(res.min())
        elif aggregate.lower().strip() == 'max':
            self.best_cycle_length = int(res.max())
        return self.best_cycle_length

    def export_jit(self, path_dir=None, drop_dt_col=True):
        if False:
            return 10
        '\n        Exporting data processing pipeline to torchscript so that it can be used without\n        Python environment. For example, when you are deploying a trained model in C++\n        and need to process input data, you can call this method to get a torchscript module\n        containing the data processing pipeline and save it in a .pt file when you finish\n        developing the model, when deploying, you can load the torchscript module from .pt\n        file and run the data processing pipeline in C++ using libtorch APIs, and the output\n        tensor can be fed into the trained model for inference.\n\n        Currently we support exporting preprocessing (scale and roll) and postprocessing (unscale)\n        to torchscript, they can do the same thing as the following code:\n\n        >>> # preprocess\n        >>> tsdata.scale(scaler, fit=False) \\\n        >>>       .roll(lookback, horizon, is_predict=True)\n        >>> preprocess_output = tsdata.to_numpy()\n        >>> # postprocess\n        >>> # "data" can be the output of model inference\n        >>> postprocess_output = tsdata.unscale_numpy(data)\n\n        Preprocessing and postprocessing will be converted to separate torchscript modules, so two\n        modules will be returned and saved.\n\n        When deploying, the compiled torchscript module can be used by:\n\n        >>> // deployment in C++\n        >>> #include <torch/torch.h>\n        >>> #include <torch/script.h>\n        >>> // create input tensor from your data\n        >>> // the data to create input tensor should have the same format as the\n        >>> // data used in developing\n        >>> torch::Tensor input_tensor = create_input_tensor(data);\n        >>> // load the module\n        >>> torch::jit::script::Module preprocessing;\n        >>> preprocessing = torch::jit::load(preprocessing_path);\n        >>> // run data preprocessing\n        >>> torch::Tensor preprocessing_output = preprocessing.forward(input_tensor).toTensor();\n        >>> // inference using your trained model\n        >>> torch::Tensor inference_output = trained_model(preprocessing_output)\n        >>> // load the postprocessing module\n        >>> torch::jit::script::Module postprocessing;\n        >>> postprocessing = torch::jit::load(postprocessing_path);\n        >>> // run postprocessing\n        >>> torch::Tensor output = postprocessing.forward(inference_output).toTensor()\n\n        Currently there are some limitations:\n            1. Please make sure the value of each column can be converted to Pytorch tensor,\n               for example, id "00" is not allowed because str can not be converted to a tensor,\n               you should use integer (0, 1, ..) as id instead of string.\n            2. Some features in tsdataset.scale and tsdataset.roll are unavailable in this\n               pipeline:\n                    a. If self.roll_additional_feature is not None, it can\'t be processed in scale\n                       and roll\n                    b. id_sensitive, time_enc and label_len parameter is not supported in roll\n            3. Users are expected to call .scale(scaler, fit=True) before calling export_jit.\n               Single roll operation is not supported for converting now.\n\n        :param path_dir: The path to save the compiled torchscript modules, default to None.\n               If set to None, you should call torch.jit.save() in your code to save the returned\n               modules; if not None, the path should be a directory, and the modules will be saved\n               at "path_dir/tsdata_preprocessing.pt" and "path_dir/tsdata_postprocessing.pt".\n        :param drop_dtcol: Whether to delete the datetime column, defaults to True. Since datetime\n               value (like "2022-12-12") can\'t be converted to Pytorch tensor, you can choose\n               different ways to workaround this. If set to True, the datetime column will be\n               deleted, then you also need to skip the datetime column when reading data from data\n               source (like csv files) in deployment environment to keep the same structure as the\n               data used in development; if set to False, the datetime column will not be deleted,\n               and you need to make sure the datetime colunm can be successfully converted to\n               Pytorch tensor when reading data in deployment environment. For example, you can set\n               each data in datetime column to an int (or other vaild types) value, since datetime\n               column is not necessary in preprocessing and postprocessing, the value can be\n               arbitrary.\n\n        :return: A tuple (preprocessing_module, postprocessing_module) containing the compiled\n                 torchscript modules.\n\n        '
        from bigdl.chronos.data.utils.export_torchscript import export_processing_to_jit, get_index
        import torch
        import os
        if drop_dt_col:
            self.df.drop(columns=self.dt_col, inplace=True)
        (id_index, target_feature_index) = get_index(self.df, self.id_col, self.target_col, self.feature_col)
        preprocessing_module = export_processing_to_jit(self.scaler, self.lookback, id_index, target_feature_index, self.scaler_index, 'preprocessing')
        postprocessing_module = export_processing_to_jit(self.scaler, self.lookback, id_index, target_feature_index, self.scaler_index, 'postprocessing')
        if path_dir:
            preprocess_path = os.path.join(path_dir, 'tsdata_preprocessing.pt')
            postprocess_path = os.path.join(path_dir, 'tsdata_postprocessing.pt')
            torch.jit.save(preprocessing_module, preprocess_path)
            torch.jit.save(postprocessing_module, postprocess_path)
        return (preprocessing_module, postprocessing_module)