from bigdl.orca.data.shard import SparkXShards
from bigdl.orca.learn.utils import dataframe_to_xshards_of_pandas_df
from bigdl.chronos.data.utils.utils import _to_list, _check_type
from bigdl.chronos.data.utils.roll import roll_timeseries_dataframe
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe
from bigdl.chronos.data.utils.split import split_timeseries_dataframe
from bigdl.chronos.data.experimental.utils import add_row, transform_to_dict
from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy
from bigdl.chronos.data.utils.feature import generate_dt_features
import pandas as pd
_DEFAULT_ID_COL_NAME = 'id'
_DEFAULT_ID_PLACEHOLDER = '0'

class XShardsTSDataset:

    def __init__(self, shards, **schema):
        if False:
            i = 10
            return i + 15
        '\n        XShardTSDataset is an abstract of time series dataset with distributed fashion.\n        Cascade call is supported for most of the transform methods.\n        XShardTSDataset will partition the dataset by id_col, which is experimental.\n        '
        self.shards = shards
        self.id_col = schema['id_col']
        self.dt_col = schema['dt_col']
        self.feature_col = schema['feature_col'].copy()
        self.target_col = schema['target_col'].copy()
        self.scaler_index = [i for i in range(len(self.target_col))]
        self.numpy_shards = None
        self._id_list = list(shards[self.id_col].unique())

    @staticmethod
    def from_xshards(shards, dt_col, target_col, id_col=None, extra_feature_col=None, with_split=False, val_ratio=0, test_ratio=0.1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize xshardtsdataset(s) from xshard pandas dataframe.\n\n        :param shards: an xshards pandas dataframe for your raw time series data.\n        :param dt_col: a str indicates the col name of datetime\n               column in the input data frame.\n        :param target_col: a str or list indicates the col name of target column\n               in the input data frame.\n        :param id_col: (optional) a str indicates the col name of dataframe id. If\n               it is not explicitly stated, then the data is interpreted as only\n               containing a single id.\n        :param extra_feature_col: (optional) a str or list indicates the col name\n               of extra feature columns that needs to predict the target column.\n        :param with_split: (optional) bool, states if we need to split the dataframe\n               to train, validation and test set. The value defaults to False.\n        :param val_ratio: (optional) float, validation ratio. Only effective when\n               with_split is set to True. The value defaults to 0.\n        :param test_ratio: (optional) float, test ratio. Only effective when with_split\n               is set to True. The value defaults to 0.1.\n\n        :return: a XShardTSDataset instance when with_split is set to False,\n                 three XShardTSDataset instances when with_split is set to True.\n\n        Create a xshardtsdataset instance by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> from bigdl.orca.data.pandas import read_csv\n        >>> shards = read_csv(csv_path)\n        >>> tsdataset = XShardsTSDataset.from_xshards(shards, dt_col="datetime",\n        >>>                                           target_col="value", id_col="id",\n        >>>                                           extra_feature_col=["extra feature 1",\n        >>>                                                              "extra feature 2"])\n        '
        _check_type(shards, 'shards', SparkXShards)
        target_col = _to_list(target_col, name='target_col')
        feature_col = _to_list(extra_feature_col, name='extra_feature_col')
        if id_col is None:
            shards = shards.transform_shard(add_row, _DEFAULT_ID_COL_NAME, _DEFAULT_ID_PLACEHOLDER)
            id_col = _DEFAULT_ID_COL_NAME
        shards = shards.partition_by(cols=id_col, num_partitions=len(shards[id_col].unique()))
        if with_split:
            tsdataset_shards = shards.transform_shard(split_timeseries_dataframe, id_col, val_ratio, test_ratio).split()
            return [XShardsTSDataset(shards=tsdataset_shards[i], id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col) for i in range(3)]
        return XShardsTSDataset(shards=shards, id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col)

    @staticmethod
    def from_sparkdf(df, dt_col, target_col, id_col=None, extra_feature_col=None, with_split=False, val_ratio=0, test_ratio=0.1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize xshardtsdataset(s) from Spark Dataframe.\n\n        :param df: an Spark DataFrame for your raw time series data.\n        :param dt_col: a str indicates the col name of datetime\n               column in the input data frame.\n        :param target_col: a str or list indicates the col name of target column\n               in the input data frame.\n        :param id_col: (optional) a str indicates the col name of dataframe id. If\n               it is not explicitly stated, then the data is interpreted as only\n               containing a single id.\n        :param extra_feature_col: (optional) a str or list indicates the col name\n               of extra feature columns that needs to predict the target column.\n        :param with_split: (optional) bool, states if we need to split the dataframe\n               to train, validation and test set. The value defaults to False.\n        :param val_ratio: (optional) float, validation ratio. Only effective when\n               with_split is set to True. The value defaults to 0.\n        :param test_ratio: (optional) float, test ratio. Only effective when with_split\n               is set to True. The value defaults to 0.1.\n\n        :return: a XShardTSDataset instance when with_split is set to False,\n                 three XShardTSDataset instances when with_split is set to True.\n\n        Create a xshardtsdataset instance by:\n\n        >>> # Here is a df example:\n        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"\n        >>> # 00        2019-01-01    1.9     1                   2\n        >>> # 01        2019-01-01    2.3     0                   9\n        >>> # 00        2019-01-02    2.4     3                   4\n        >>> # 01        2019-01-02    2.6     0                   2\n        >>> df = <pyspark.sql.dataframe.DataFrame>\n        >>> tsdataset = XShardsTSDataset.from_xshards(df, dt_col="datetime",\n        >>>                                           target_col="value", id_col="id",\n        >>>                                           extra_feature_col=["extra feature 1",\n        >>>                                                              "extra feature 2"])\n        '
        from pyspark.sql.dataframe import DataFrame
        _check_type(df, 'df', DataFrame)
        target_col = _to_list(target_col, name='target_col')
        feature_col = _to_list(extra_feature_col, name='extra_feature_col')
        all_col = target_col + feature_col + _to_list(id_col, name='id_col') + [dt_col]
        shards = dataframe_to_xshards_of_pandas_df(df, feature_cols=all_col, label_cols=None, accept_str_col=False)
        if id_col is None:
            shards = shards.transform_shard(add_row, _DEFAULT_ID_COL_NAME, _DEFAULT_ID_PLACEHOLDER)
            id_col = _DEFAULT_ID_COL_NAME
        shards = shards.partition_by(cols=id_col, num_partitions=len(shards[id_col].unique()))
        if with_split:
            tsdataset_shards = shards.transform_shard(split_timeseries_dataframe, id_col, val_ratio, test_ratio).split()
            return [XShardsTSDataset(shards=tsdataset_shards[i], id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col) for i in range(3)]
        return XShardsTSDataset(shards=shards, id_col=id_col, dt_col=dt_col, target_col=target_col, feature_col=feature_col)

    def roll(self, lookback, horizon, feature_col=None, target_col=None, id_sensitive=False):
        if False:
            print('Hello World!')
        "\n        Sampling by rolling for machine learning/deep learning models.\n\n        :param lookback: int, lookback value.\n        :param horizon: int or list,\n               if `horizon` is an int, we will sample `horizon` step\n               continuously after the forecasting point.\n               if `horizon` is a list, we will sample discretely according\n               to the input list.\n               specially, when `horizon` is set to 0, ground truth will be generated as None.\n        :param feature_col: str or list, indicates the feature col name. Default to None,\n               where we will take all available feature in rolling.\n        :param target_col: str or list, indicates the target col name. Default to None,\n               where we will take all target in rolling. it should be a subset of target_col\n               you used to initialize the xshardtsdataset.\n        :param id_sensitive: bool,\n               |if `id_sensitive` is False, we will rolling on each id's sub dataframe\n               |and fuse the sampings.\n               |The shape of rolling will be\n               |x: (num_sample, lookback, num_feature_col + num_target_col)\n               |y: (num_sample, horizon, num_target_col)\n               |where num_sample is the summation of sample number of each dataframe\n               |if `id_sensitive` is True, we have not implement this currently.\n\n        :return: the xshardtsdataset instance.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if id_sensitive:
            invalidInputError(False, 'id_sensitive option has not been implemented.')
        feature_col = _to_list(feature_col, 'feature_col') if feature_col is not None else self.feature_col
        target_col = _to_list(target_col, 'target_col') if target_col is not None else self.target_col
        self.numpy_shards = self.shards.transform_shard(roll_timeseries_dataframe, None, lookback, horizon, feature_col, target_col, self.id_col, 0, True)
        self.scaler_index = [self.target_col.index(target_col[i]) for i in range(len(target_col))]
        return self

    def scale(self, scaler, fit=True):
        if False:
            while True:
                i = 10
        '\n        Scale the time series dataset\'s feature column and target column.\n\n        :param scaler: a dictionary of scaler instance, where keys are id name\n               and values are corresponding scaler instance. e.g. if you have\n               2 ids called "id1" and "id2", a legal scaler input can be\n               {"id1": StandardScaler(), "id2": StandardScaler()}\n        :param fit: if we need to fit the scaler. Typically, the value should\n               be set to True for training set, while False for validation and\n               test set. The value is defaulted to True.\n\n        :return: the xshardtsdataset instance.\n\n        Assume there is a training set tsdata and a test set tsdata_test.\n        scale() should be called first on training set with default value fit=True,\n        then be called on test set with the same scaler and fit=False.\n\n        >>> from sklearn.preprocessing import StandardScaler\n        >>> scaler = {"id1": StandardScaler(), "id2": StandardScaler()}\n        >>> tsdata.scale(scaler, fit=True)\n        >>> tsdata_test.scale(scaler, fit=False)\n        '

        def _fit(df, id_col, scaler, feature_col, target_col):
            if False:
                print('Hello World!')
            '\n            This function is used to fit scaler dictionary on each shard.\n            returns a dictionary of id-scaler pair for each shard.\n\n            Note: this function will not transform the shard.\n            '
            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            df[target_col + feature_col] = scaler_for_this_id.fit(df[target_col + feature_col])
            return {id_col: df[id_col].iloc[0], 'scaler': scaler_for_this_id}

        def _transform(df, id_col, scaler, feature_col, target_col):
            if False:
                i = 10
                return i + 15
            '\n            This function is used to transform the shard by fitted scaler.\n\n            Note: this function will not fit the scaler.\n            '
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.common import invalidInputError
            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            invalidInputError(not check_is_fitted(scaler_for_this_id), 'scaler is not fitted. When calling scale for the first time, you need to set fit=True.')
            df[target_col + feature_col] = scaler_for_this_id.transform(df[target_col + feature_col])
            return df
        if fit:
            self.shards_scaler = self.shards.transform_shard(_fit, self.id_col, scaler, self.feature_col, self.target_col)
            self.scaler_dict = self.shards_scaler.collect()
            self.scaler_dict = {sc[self.id_col]: sc['scaler'] for sc in self.scaler_dict}
            scaler.update(self.scaler_dict)
            self.shards = self.shards.transform_shard(_transform, self.id_col, self.scaler_dict, self.feature_col, self.target_col)
        else:
            self.scaler_dict = scaler
            self.shards = self.shards.transform_shard(_transform, self.id_col, self.scaler_dict, self.feature_col, self.target_col)
        return self

    def unscale(self):
        if False:
            while True:
                i = 10
        "\n        Unscale the time series dataset's feature column and target column.\n\n        :return: the xshardtsdataset instance.\n        "

        def _inverse_transform(df, id_col, scaler, feature_col, target_col):
            if False:
                i = 10
                return i + 15
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.common import invalidInputError
            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            invalidInputError(not check_is_fitted(scaler_for_this_id), 'scaler is not fitted. When calling scale for the first time, you need to set fit=True.')
            df[target_col + feature_col] = scaler_for_this_id.inverse_transform(df[target_col + feature_col])
            return df
        self.shards = self.shards.transform_shard(_inverse_transform, self.id_col, self.scaler_dict, self.feature_col, self.target_col)
        return self

    def gen_dt_feature(self, features):
        if False:
            print('Hello World!')
        '\n        Generate datetime feature(s) for each record.\n        :param features: list, states which feature(s) will be generated.\n                The list should contain the features you want to generate.\n                A table of all datatime features and their description is listed below.\n\n        | "MINUTE": The minute of the time stamp.\n        | "DAY": The day of the time stamp.\n        | "DAYOFYEAR": The ordinal day of the year of the time stamp.\n        | "HOUR": The hour of the time stamp.\n        | "WEEKDAY": The day of the week of the time stamp, Monday=0, Sunday=6.\n        | "WEEKOFYEAR": The ordinal week of the year of the time stamp.\n        | "MONTH": The month of the time stamp.\n        | "YEAR": The year of the time stamp.\n        | "IS_AWAKE": Bool value indicating whether it belongs to awake hours for the time stamp,\n        | True for hours between 6A.M. and 1A.M.\n        | "IS_BUSY_HOURS": Bool value indicating whether it belongs to busy hours for the time\n        | stamp, True for hours between 7A.M. and 10A.M. and hours between 4P.M. and 8P.M.\n        | "IS_WEEKEND": Bool value indicating whether it belongs to weekends for the time stamp,\n        | True for Saturdays and Sundays.\n\n        :return: the xshards instance.\n        '
        features_generated = []
        self.shards = self.shards.transform_shard(generate_dt_features, self.dt_col, features, None, None, features_generated)
        features_generated = [fe for fe in features if fe not in self.target_col + [self.dt_col, self.id_col]]
        self.feature_col += features_generated
        return self

    def unscale_xshards(self, data, key=None):
        if False:
            print('Hello World!')
        '\n        Unscale the time series forecaster\'s numpy prediction result/ground truth.\n\n        :param data: xshards same with self.numpy_xshards.\n        :param key: str, \'y\' or \'prediction\', default to \'y\'. if no "prediction"\n        or "y" return an error and require our users to input a key. if key is\n        None, key will be set \'prediction\'.\n\n        :return: the unscaled xshardtsdataset instance.\n        '
        from bigdl.nano.utils.common import invalidInputError

        def _inverse_transform(data, scaler, scaler_index, key):
            if False:
                while True:
                    i = 10
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.common import invalidInputError
            id = data['id'][0, 0]
            scaler_for_this_id = scaler[id]
            invalidInputError(not check_is_fitted(scaler_for_this_id), 'scaler is not fitted. When calling scale for the first time, you need to set fit=True.')
            return unscale_timeseries_numpy(data[key], scaler_for_this_id, scaler_index)
        if key is None:
            key = 'prediction'
        invalidInputError(key in {'y', 'prediction'}, "key is not in {'y', 'prediction'}, please input the correct key.")
        return data.transform_shard(_inverse_transform, self.scaler_dict, self.scaler_index, key)

    def impute(self, mode='last', const_num=0):
        if False:
            while True:
                i = 10
        '\n        Impute the tsdataset by imputing each univariate time series\n        distinguished by id_col and feature_col.\n\n        :param mode: imputation mode, select from "last", "const" or "linear".\n\n            "last": impute by propagating the last non N/A number to its following N/A.\n            if there is no non N/A number ahead, 0 is filled instead.\n\n            "const": impute by a const value input by user.\n\n            "linear": impute by linear interpolation.\n        :param const_num:  indicates the const number to fill, which is only effective when mode\n            is set to "const".\n\n        :return: the tsdataset instance.\n        '

        def df_reset_index(df):
            if False:
                print('Hello World!')
            df.reset_index(drop=True, inplace=True)
            return df
        self.shards = self.shards.transform_shard(impute_timeseries_dataframe, self.dt_col, mode, const_num)
        self.shards = self.shards.transform_shard(df_reset_index)
        return self

    def to_xshards(self, partition_num=None):
        if False:
            print('Hello World!')
        "\n        Export rolling result in form of a dict of numpy ndarray {'x': ..., 'y': ..., 'id': ...},\n        where value for 'x' and 'y' are 3-dim numpy ndarray and value for 'id' is 2-dim ndarray\n        with shape (batch_size, 1)\n\n        :param partition_num: how many partition you would like to split your data.\n\n        :return: a 3-element dict xshard. each value is a 3d numpy ndarray. The ndarray\n                 is casted to float32. Default to None which will partition according\n                 to id.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if self.numpy_shards is None:
            invalidInputError(False, "Please call 'roll' method before transform a XshardsTSDataset to numpy ndarray!")
        if partition_num is None:
            return self.numpy_shards.transform_shard(transform_to_dict)
        else:
            return self.numpy_shards.transform_shard(transform_to_dict).repartition(partition_num)