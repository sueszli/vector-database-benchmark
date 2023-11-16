"""System tests for Google BigQuery hooks"""
from __future__ import annotations
import pytest
from airflow.providers.google.cloud.hooks import bigquery as hook
from tests.providers.google.cloud.utils.gcp_authenticator import GCP_BIGQUERY_KEY
from tests.test_utils.gcp_system_helpers import GoogleSystemTest

@pytest.mark.system('google.cloud')
@pytest.mark.credential_file(GCP_BIGQUERY_KEY)
class TestBigQueryDataframeResultsSystem(GoogleSystemTest):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.instance = hook.BigQueryHook()

    def test_output_is_dataframe_with_valid_query(self):
        if False:
            while True:
                i = 10
        import pandas as pd
        df = self.instance.get_pandas_df('select 1')
        assert isinstance(df, pd.DataFrame)

    def test_throws_exception_with_invalid_query(self):
        if False:
            print('Hello World!')
        with pytest.raises(Exception) as ctx:
            self.instance.get_pandas_df('from `1`')
        assert 'Reason: ' in str(ctx.value), ''

    def test_succeeds_with_explicit_legacy_query(self):
        if False:
            i = 10
            return i + 15
        df = self.instance.get_pandas_df('select 1', dialect='legacy')
        assert df.iloc(0)[0][0] == 1

    def test_succeeds_with_explicit_std_query(self):
        if False:
            print('Hello World!')
        df = self.instance.get_pandas_df('select * except(b) from (select 1 a, 2 b)', dialect='standard')
        assert df.iloc(0)[0][0] == 1

    def test_throws_exception_with_incompatible_syntax(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(Exception) as ctx:
            self.instance.get_pandas_df('select * except(b) from (select 1 a, 2 b)', dialect='legacy')
        assert 'Reason: ' in str(ctx.value), ''