from __future__ import annotations
from unittest import mock
from airflow.providers.google.cloud.transfers.bigquery_to_postgres import BigQueryToPostgresOperator
TASK_ID = 'test-bq-create-table-operator'
TEST_DATASET = 'test-dataset'
TEST_TABLE_ID = 'test-table-id'
TEST_DAG_ID = 'test-bigquery-operators'

class TestBigQueryToPostgresOperator:

    @mock.patch('airflow.providers.google.cloud.transfers.bigquery_to_sql.BigQueryHook')
    def test_execute_good_request_to_bq(self, mock_hook):
        if False:
            return 10
        destination_table = 'table'
        operator = BigQueryToPostgresOperator(task_id=TASK_ID, dataset_table=f'{TEST_DATASET}.{TEST_TABLE_ID}', target_table_name=destination_table, replace=False)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.list_rows.assert_called_once_with(dataset_id=TEST_DATASET, table_id=TEST_TABLE_ID, max_results=1000, selected_fields=None, start_index=0)