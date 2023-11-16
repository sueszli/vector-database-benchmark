from unittest import mock
import pytest
from dagster import configured, job, op
from dagster_gcp.gcs.file_manager import GCSFileHandle, GCSFileManager
from dagster_gcp.gcs.resources import gcs_file_manager
from google.cloud import storage

@pytest.mark.integration
def test_gcs_file_manager_write():
    if False:
        print('Hello World!')
    gcs_mock = mock.MagicMock()
    file_manager = GCSFileManager(storage.client.Client(), 'some-bucket', 'some-key')
    file_manager._client = gcs_mock
    foo_bytes = b'foo'
    file_handle = file_manager.write_data(foo_bytes)
    assert isinstance(file_handle, GCSFileHandle)
    assert file_handle.gcs_bucket == 'some-bucket'
    assert file_handle.gcs_key.startswith('some-key/')
    assert gcs_mock.bucket().blob().upload_from_file.call_count == 1
    file_handle = file_manager.write_data(foo_bytes, ext='foo', key='test')
    assert isinstance(file_handle, GCSFileHandle)
    assert file_handle.gcs_bucket == 'some-bucket'
    assert file_handle.gcs_key.startswith('some-key/')
    assert file_handle.gcs_key.find('test') != -1
    assert file_handle.gcs_key[-4:] == '.foo'
    assert gcs_mock.bucket().blob().upload_from_file.call_count == 2

@mock.patch('dagster_gcp.gcs.resources.storage.client.Client')
@mock.patch('dagster_gcp.gcs.resources.GCSFileManager')
def test_gcs_file_manger_resource(MockGCSFileManager, mock_storage_client_Client):
    if False:
        for i in range(10):
            print('nop')
    did_it_run = dict(it_ran=False)
    resource_config = {'project': 'some-project', 'gcs_bucket': 'some-bucket', 'gcs_prefix': 'some-prefix'}

    @op(required_resource_keys={'file_manager'})
    def test_op(context):
        if False:
            while True:
                i = 10
        assert context.resources.file_manager == MockGCSFileManager.return_value
        MockGCSFileManager.assert_called_once_with(client=mock_storage_client_Client.return_value, gcs_bucket=resource_config['gcs_bucket'], gcs_base_key=resource_config['gcs_prefix'])
        mock_storage_client_Client.assert_called_once_with(project=resource_config['project'])
        did_it_run['it_ran'] = True

    @job(resource_defs={'file_manager': configured(gcs_file_manager)(resource_config)})
    def test_job():
        if False:
            while True:
                i = 10
        test_op()
    test_job.execute_in_process()
    assert did_it_run['it_ran']