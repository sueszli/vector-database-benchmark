from __future__ import annotations
from unittest import mock
from airflow.providers.presto.transfers.gcs_to_presto import GCSToPrestoOperator
BUCKET = 'source_bucket'
PATH = 'path/to/file.csv'
GCP_CONN_ID = 'test_gcp'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']
PRESTO_CONN_ID = 'test_presto'
PRESTO_TABLE = 'test_table'
TASK_ID = 'test_gcs_to_presto'
SCHEMA_FIELDS = ['colA', 'colB', 'colC']
SCHEMA_JSON = 'path/to/file.json'

class TestGCSToPrestoOperator:

    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.PrestoHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.GCSHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.NamedTemporaryFile')
    def test_execute_without_schema(self, mock_tempfile, mock_gcs_hook, mock_presto_hook):
        if False:
            for i in range(10):
                print('nop')
        filename = 'file://97g23r'
        file_handle = mock.MagicMock()
        mock_tempfile.return_value.__enter__.return_value = file_handle
        mock_tempfile.return_value.__enter__.return_value.name = filename
        op = GCSToPrestoOperator(task_id=TASK_ID, source_bucket=BUCKET, source_object=PATH, presto_table=PRESTO_TABLE, presto_conn_id=PRESTO_CONN_ID, gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        op.execute(None)
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_presto_hook.assert_called_once_with(presto_conn_id=PRESTO_CONN_ID)
        mock_download = mock_gcs_hook.return_value.download
        mock_download.assert_called_once_with(bucket_name=BUCKET, object_name=PATH, filename=filename)
        mock_insert = mock_presto_hook.return_value.insert_rows
        mock_insert.assert_called_once()

    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.PrestoHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.GCSHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.NamedTemporaryFile')
    def test_execute_schema_fields(self, mock_tempfile, mock_gcs_hook, mock_presto_hook):
        if False:
            i = 10
            return i + 15
        filename = 'file://97g23r'
        file_handle = mock.MagicMock()
        mock_tempfile.return_value.__enter__.return_value = file_handle
        mock_tempfile.return_value.__enter__.return_value.name = filename
        op = GCSToPrestoOperator(task_id=TASK_ID, source_bucket=BUCKET, source_object=PATH, presto_table=PRESTO_TABLE, presto_conn_id=PRESTO_CONN_ID, gcp_conn_id=GCP_CONN_ID, schema_fields=SCHEMA_FIELDS, impersonation_chain=IMPERSONATION_CHAIN)
        op.execute(None)
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_presto_hook.assert_called_once_with(presto_conn_id=PRESTO_CONN_ID)
        mock_download = mock_gcs_hook.return_value.download
        mock_download.assert_called_once_with(bucket_name=BUCKET, object_name=PATH, filename=filename)
        mock_insert = mock_presto_hook.return_value.insert_rows
        mock_insert.assert_called_once()

    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.json.loads')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.PrestoHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.GCSHook')
    @mock.patch('airflow.providers.presto.transfers.gcs_to_presto.NamedTemporaryFile')
    def test_execute_schema_json(self, mock_tempfile, mock_gcs_hook, mock_presto_hook, mock_json_loader):
        if False:
            return 10
        filename = 'file://97g23r'
        file_handle = mock.MagicMock()
        mock_tempfile.return_value.__enter__.return_value = file_handle
        mock_tempfile.return_value.__enter__.return_value.name = filename
        mock_json_loader.return_value = SCHEMA_FIELDS
        op = GCSToPrestoOperator(task_id=TASK_ID, source_bucket=BUCKET, source_object=PATH, presto_table=PRESTO_TABLE, presto_conn_id=PRESTO_CONN_ID, gcp_conn_id=GCP_CONN_ID, schema_object=SCHEMA_JSON, impersonation_chain=IMPERSONATION_CHAIN)
        op.execute(None)
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=GCP_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        mock_presto_hook.assert_called_once_with(presto_conn_id=PRESTO_CONN_ID)
        mock_download = mock_gcs_hook.return_value.download
        assert mock_download.call_count == 2
        mock_insert = mock_presto_hook.return_value.insert_rows
        mock_insert.assert_called_once()