import pytest
import pandas as pd
import numpy as np
from unittest import TestCase
from bigdl.chronos.data.utils.feature import generate_dt_features, generate_global_features
from bigdl.chronos.utils import LazyImport
tsfresh = LazyImport('tsfresh')
from ... import op_torch, op_tf2, op_diff_set_all

@op_torch
@op_tf2
class TestFeature(TestCase):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        pass

    def teardown_method(self, method):
        if False:
            print('Hello World!')
        pass

    def test_generate_dt_features(self):
        if False:
            while True:
                i = 10
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({'datetime': dates, 'values': data[:, 0], 'A': data[:, 1], 'B': data[:, 2]})
        df = generate_dt_features(df, dt_col='datetime', features='auto', one_hot_features=None, freq=pd.Timedelta('1D'), features_generated=[])
        assert set(df.columns) == {'DAY', 'IS_WEEKEND', 'WEEKDAY', 'MONTH', 'DAYOFYEAR', 'WEEKOFYEAR', 'A', 'B', 'YEAR', 'values', 'datetime'}

    @op_diff_set_all
    def test_gen_global_feature_single_id(self):
        if False:
            return 10
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({'datetime': dates, 'values': data[:, 0], 'A': data[:, 1], 'B': data[:, 2], 'id': ['00'] * 8})
        from tsfresh.feature_extraction import MinimalFCParameters
        for params in [MinimalFCParameters()]:
            (output_df, _) = generate_global_features(input_df=df, column_id='id', column_sort='datetime', default_fc_parameters=params)
            assert 'datetime' in output_df.columns
            assert 'values' in output_df.columns
            assert 'A' in output_df.columns
            assert 'B' in output_df.columns
            assert 'id' in output_df.columns
            for col in output_df.columns:
                if col in ['datetime', 'values', 'A', 'B', 'id']:
                    continue
                assert len(set(output_df[col])) == 1
                assert output_df[col].isna().sum() == 0

    @op_diff_set_all
    def test_gen_global_feature_multi_id(self):
        if False:
            i = 10
            return i + 15
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({'datetime': dates, 'values': data[:, 0], 'A': data[:, 1], 'B': data[:, 2], 'id': ['00'] * 4 + ['01'] * 4})
        from tsfresh.feature_extraction import MinimalFCParameters
        for params in [MinimalFCParameters()]:
            (output_df, _) = generate_global_features(input_df=df, column_id='id', column_sort='datetime', default_fc_parameters=params)
            assert 'datetime' in output_df.columns
            assert 'values' in output_df.columns
            assert 'A' in output_df.columns
            assert 'B' in output_df.columns
            assert 'id' in output_df.columns
            for col in output_df.columns:
                if col in ['datetime', 'values', 'A', 'B', 'id']:
                    continue
                assert len(set(output_df[output_df['id'] == '00'][col])) == 1
                assert len(set(output_df[output_df['id'] == '01'][col])) == 1
                assert output_df[output_df['id'] == '00'][col].isna().sum() == 0
                assert output_df[output_df['id'] == '01'][col].isna().sum() == 0