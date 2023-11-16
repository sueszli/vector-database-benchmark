from __future__ import annotations
from unittest import mock
from airflow.providers.google.cloud.transfers.adls_to_gcs import ADLSToGCSOperator
TASK_ID = 'test-adls-gcs-operator'
ADLS_PATH_1 = '*'
GCS_PATH = 'gs://test/'
MOCK_FILES = ['test/TEST1.csv', 'test/TEST2.csv', 'test/path/TEST3.csv', 'test/path/PARQUET.parquet', 'test/path/PIC.png']
AZURE_CONN_ID = 'azure_data_lake_default'
GCS_CONN_ID = 'google_cloud_default'
IMPERSONATION_CHAIN = ['ACCOUNT_1', 'ACCOUNT_2', 'ACCOUNT_3']

class TestAdlsToGoogleCloudStorageOperator:

    def test_init(self):
        if False:
            return 10
        'Test AdlsToGoogleCloudStorageOperator instance is properly initialized.'
        operator = ADLSToGCSOperator(task_id=TASK_ID, src_adls=ADLS_PATH_1, dest_gcs=GCS_PATH, replace=False, azure_data_lake_conn_id=AZURE_CONN_ID, gcp_conn_id=GCS_CONN_ID)
        assert operator.task_id == TASK_ID
        assert operator.src_adls == ADLS_PATH_1
        assert operator.dest_gcs == GCS_PATH
        assert operator.replace is False
        assert operator.gcp_conn_id == GCS_CONN_ID
        assert operator.azure_data_lake_conn_id == AZURE_CONN_ID

    @mock.patch('airflow.providers.google.cloud.transfers.adls_to_gcs.AzureDataLakeHook')
    @mock.patch('airflow.providers.microsoft.azure.operators.adls.AzureDataLakeHook')
    @mock.patch('airflow.providers.google.cloud.transfers.adls_to_gcs.GCSHook')
    def test_execute(self, gcs_mock_hook, adls_one_mock_hook, adls_two_mock_hook):
        if False:
            return 10
        'Test the execute function when the run is successful.'
        operator = ADLSToGCSOperator(task_id=TASK_ID, src_adls=ADLS_PATH_1, dest_gcs=GCS_PATH, replace=False, azure_data_lake_conn_id=AZURE_CONN_ID, gcp_conn_id=GCS_CONN_ID, google_impersonation_chain=IMPERSONATION_CHAIN)
        adls_one_mock_hook.return_value.list.return_value = MOCK_FILES
        adls_two_mock_hook.return_value.list.return_value = MOCK_FILES
        uploaded_files = operator.execute(None)
        gcs_mock_hook.return_value.upload.assert_has_calls([mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/PARQUET.parquet', gzip=False), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/TEST3.csv', gzip=False), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/PIC.png', gzip=False), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/TEST1.csv', gzip=False), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/TEST2.csv', gzip=False)], any_order=True)
        adls_one_mock_hook.assert_called_once_with(azure_data_lake_conn_id=AZURE_CONN_ID)
        adls_two_mock_hook.assert_called_once_with(azure_data_lake_conn_id=AZURE_CONN_ID)
        gcs_mock_hook.assert_called_once_with(gcp_conn_id=GCS_CONN_ID, impersonation_chain=IMPERSONATION_CHAIN)
        assert sorted(MOCK_FILES) == sorted(uploaded_files)

    @mock.patch('airflow.providers.google.cloud.transfers.adls_to_gcs.AzureDataLakeHook')
    @mock.patch('airflow.providers.microsoft.azure.operators.adls.AzureDataLakeHook')
    @mock.patch('airflow.providers.google.cloud.transfers.adls_to_gcs.GCSHook')
    def test_execute_with_gzip(self, gcs_mock_hook, adls_one_mock_hook, adls_two_mock_hook):
        if False:
            return 10
        'Test the execute function when the run is successful.'
        operator = ADLSToGCSOperator(task_id=TASK_ID, src_adls=ADLS_PATH_1, dest_gcs=GCS_PATH, replace=False, azure_data_lake_conn_id=AZURE_CONN_ID, gcp_conn_id=GCS_CONN_ID, gzip=True)
        adls_one_mock_hook.return_value.list.return_value = MOCK_FILES
        adls_two_mock_hook.return_value.list.return_value = MOCK_FILES
        uploaded_files = operator.execute(None)
        gcs_mock_hook.return_value.upload.assert_has_calls([mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/PARQUET.parquet', gzip=True), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/TEST3.csv', gzip=True), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/path/PIC.png', gzip=True), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/TEST1.csv', gzip=True), mock.call(bucket_name='test', filename=mock.ANY, object_name='test/TEST2.csv', gzip=True)], any_order=True)
        assert sorted(MOCK_FILES) == sorted(uploaded_files)