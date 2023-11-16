from __future__ import annotations
from unittest import mock
import pandas as pd
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.slack.transfers.base_sql_to_slack import BaseSqlToSlackOperator

class TestBaseSqlToSlackOperator:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.default_op_kwargs = {'sql': 'SELECT 1', 'sql_conn_id': 'test-sql-conn-id', 'sql_hook_params': None, 'parameters': None}

    def test_execute_not_implemented(self):
        if False:
            return 10
        'Test that no base implementation for ``BaseSqlToSlackOperator.execute()``.'
        op = BaseSqlToSlackOperator(task_id='test_base_not_implements', **self.default_op_kwargs)
        with pytest.raises(NotImplementedError):
            op.execute(mock.MagicMock())

    @mock.patch('airflow.providers.common.sql.operators.sql.BaseHook.get_connection')
    @mock.patch('airflow.models.connection.Connection.get_hook')
    @pytest.mark.parametrize('conn_type', ['postgres', 'snowflake'])
    @pytest.mark.parametrize('sql_hook_params', [None, {'foo': 'bar'}])
    def test_get_hook(self, mock_get_hook, mock_get_conn, conn_type, sql_hook_params):
        if False:
            return 10

        class SomeDummyHook:
            """Hook which implements ``get_pandas_df`` method"""

            def get_pandas_df(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        expected_hook = SomeDummyHook()
        mock_get_conn.return_value = Connection(conn_id=f'test_connection_{conn_type}', conn_type=conn_type)
        mock_get_hook.return_value = expected_hook
        op_kwargs = {**self.default_op_kwargs, 'sql_hook_params': sql_hook_params}
        op = BaseSqlToSlackOperator(task_id='test_get_hook', **op_kwargs)
        hook = op._get_hook()
        mock_get_hook.assert_called_once_with(hook_params=sql_hook_params)
        assert hook == expected_hook

    @mock.patch('airflow.providers.common.sql.operators.sql.BaseHook.get_connection')
    @mock.patch('airflow.models.connection.Connection.get_hook')
    def test_get_not_supported_hook(self, mock_get_hook, mock_get_conn):
        if False:
            for i in range(10):
                print('nop')

        class SomeDummyHook:
            """Hook which not implemented ``get_pandas_df`` method"""
        mock_get_conn.return_value = Connection(conn_id='test_connection', conn_type='test_connection')
        mock_get_hook.return_value = SomeDummyHook()
        op = BaseSqlToSlackOperator(task_id='test_get_not_supported_hook', **self.default_op_kwargs)
        error_message = 'This hook is not supported. The hook class must have get_pandas_df method\\.'
        with pytest.raises(AirflowException, match=error_message):
            op._get_hook()

    @mock.patch('airflow.providers.slack.transfers.sql_to_slack.BaseSqlToSlackOperator._get_hook')
    @pytest.mark.parametrize('sql', ['SELECT 42', 'SELECT 1 FROM DUMMY WHERE col = ?'])
    @pytest.mark.parametrize('parameters', [None, {'col': 'spam-egg'}])
    def test_get_query_results(self, mock_op_get_hook, sql, parameters):
        if False:
            while True:
                i = 10
        test_df = pd.DataFrame({'a': '1', 'b': '2'}, index=[0, 1])
        mock_get_pandas_df = mock.MagicMock(return_value=test_df)
        mock_hook = mock.MagicMock()
        mock_hook.get_pandas_df = mock_get_pandas_df
        mock_op_get_hook.return_value = mock_hook
        op_kwargs = {**self.default_op_kwargs, 'sql': sql, 'parameters': parameters}
        op = BaseSqlToSlackOperator(task_id='test_get_query_results', **op_kwargs)
        df = op._get_query_results()
        mock_get_pandas_df.assert_called_once_with(sql, parameters=parameters)
        assert df is test_df