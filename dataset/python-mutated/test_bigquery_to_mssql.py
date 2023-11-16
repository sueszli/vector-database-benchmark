from __future__ import annotations
from unittest import mock
import pytest
try:
    from airflow.providers.google.cloud.transfers.bigquery_to_mssql import BigQueryToMsSqlOperator
except ImportError:
    pytest.skip('MSSQL not available', allow_module_level=True)
TASK_ID = 'test-bq-create-table-operator'
TEST_PROJECT_ID = 'test-project'
TEST_DATASET = 'test-dataset'
TEST_TABLE_ID = 'test-table-id'
TEST_DAG_ID = 'test-bigquery-operators'

@pytest.mark.backend('mssql')
class TestBigQueryToMsSqlOperator:

    @mock.patch('airflow.providers.google.cloud.transfers.bigquery_to_sql.BigQueryHook')
    def test_execute_good_request_to_bq(self, mock_hook):
        if False:
            print('Hello World!')
        destination_table = 'table'
        operator = BigQueryToMsSqlOperator(task_id=TASK_ID, source_project_dataset_table=f'{TEST_PROJECT_ID}.{TEST_DATASET}.{TEST_TABLE_ID}', target_table_name=destination_table, replace=False)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.list_rows.assert_called_once_with(dataset_id=TEST_DATASET, table_id=TEST_TABLE_ID, max_results=1000, selected_fields=None, start_index=0)