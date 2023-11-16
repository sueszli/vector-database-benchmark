from __future__ import annotations
from unittest import mock
from airflow.providers.google.cloud.transfers.bigquery_to_bigquery import BigQueryToBigQueryOperator
BQ_HOOK_PATH = 'airflow.providers.google.cloud.transfers.bigquery_to_bigquery.BigQueryHook'
TASK_ID = 'test-bq-create-table-operator'
TEST_DATASET = 'test-dataset'
TEST_TABLE_ID = 'test-table-id'

class TestBigQueryToBigQueryOperator:

    @mock.patch(BQ_HOOK_PATH)
    def test_execute_without_location_should_execute_successfully(self, mock_hook):
        if False:
            while True:
                i = 10
        source_project_dataset_tables = f'{TEST_DATASET}.{TEST_TABLE_ID}'
        destination_project_dataset_table = f"{TEST_DATASET + '_new'}.{TEST_TABLE_ID}"
        write_disposition = 'WRITE_EMPTY'
        create_disposition = 'CREATE_IF_NEEDED'
        labels = {'k1': 'v1'}
        encryption_configuration = {'key': 'kk'}
        operator = BigQueryToBigQueryOperator(task_id=TASK_ID, source_project_dataset_tables=source_project_dataset_tables, destination_project_dataset_table=destination_project_dataset_table, write_disposition=write_disposition, create_disposition=create_disposition, labels=labels, encryption_configuration=encryption_configuration)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.run_copy.assert_called_once_with(source_project_dataset_tables=source_project_dataset_tables, destination_project_dataset_table=destination_project_dataset_table, write_disposition=write_disposition, create_disposition=create_disposition, labels=labels, encryption_configuration=encryption_configuration)

    @mock.patch(BQ_HOOK_PATH)
    def test_execute_single_regional_location_should_execute_successfully(self, mock_hook):
        if False:
            while True:
                i = 10
        source_project_dataset_tables = f'{TEST_DATASET}.{TEST_TABLE_ID}'
        destination_project_dataset_table = f"{TEST_DATASET + '_new'}.{TEST_TABLE_ID}"
        write_disposition = 'WRITE_EMPTY'
        create_disposition = 'CREATE_IF_NEEDED'
        labels = {'k1': 'v1'}
        location = 'us-central1'
        encryption_configuration = {'key': 'kk'}
        mock_hook.return_value.run_copy.return_value = 'job-id'
        operator = BigQueryToBigQueryOperator(task_id=TASK_ID, source_project_dataset_tables=source_project_dataset_tables, destination_project_dataset_table=destination_project_dataset_table, write_disposition=write_disposition, create_disposition=create_disposition, labels=labels, encryption_configuration=encryption_configuration, location=location)
        operator.execute(context=mock.MagicMock())
        mock_hook.return_value.get_job.assert_called_once_with(job_id='job-id', location=location)