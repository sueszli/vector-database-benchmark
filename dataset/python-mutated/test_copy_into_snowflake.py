from __future__ import annotations
from unittest import mock
from airflow.providers.snowflake.transfers.copy_into_snowflake import CopyFromExternalStageToSnowflakeOperator

class TestCopyFromExternalStageToSnowflake:

    @mock.patch('airflow.providers.snowflake.transfers.copy_into_snowflake.SnowflakeHook')
    def test_execute(self, mock_hook):
        if False:
            while True:
                i = 10
        CopyFromExternalStageToSnowflakeOperator(table='table', file_format='CSV', stage='stage', prefix='prefix', columns_array=['col1, col2'], files=['file1.csv', 'file2.csv'], pattern='*.csv', warehouse='warehouse', database='database', role='role', schema='schema', authenticator='authenticator', copy_options='copy_options', validation_mode='validation_mode', task_id='test').execute(None)
        mock_hook.assert_called_once_with(snowflake_conn_id='snowflake_default', warehouse='warehouse', database='database', role='role', schema='schema', authenticator='authenticator', session_parameters=None)
        sql = "\n        COPY INTO schema.table(col1, col2)\n             FROM  @stage/prefix\n        FILES=('file1.csv','file2.csv')\n        PATTERN='*.csv'\n        FILE_FORMAT=CSV\n        copy_options\n        validation_mode\n        "
        mock_hook.return_value.run.assert_called_once_with(sql=sql, autocommit=True)