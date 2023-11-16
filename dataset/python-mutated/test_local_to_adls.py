from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.microsoft.azure.transfers.local_to_adls import LocalFilesystemToADLSOperator
TASK_ID = 'test-adls-upload-operator'
LOCAL_PATH = 'test/*'
BAD_LOCAL_PATH = 'test/**'
REMOTE_PATH = 'TEST-DIR'

class TestADLSUploadOperator:

    @mock.patch('airflow.providers.microsoft.azure.transfers.local_to_adls.AzureDataLakeHook')
    def test_execute_success(self, mock_hook):
        if False:
            print('Hello World!')
        operator = LocalFilesystemToADLSOperator(task_id=TASK_ID, local_path=LOCAL_PATH, remote_path=REMOTE_PATH)
        operator.execute(None)
        mock_hook.return_value.upload_file.assert_called_once_with(local_path=LOCAL_PATH, remote_path=REMOTE_PATH, nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

    @mock.patch('airflow.providers.microsoft.azure.transfers.local_to_adls.AzureDataLakeHook')
    def test_execute_raises_for_bad_glob_val(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        operator = LocalFilesystemToADLSOperator(task_id=TASK_ID, local_path=BAD_LOCAL_PATH, remote_path=REMOTE_PATH)
        with pytest.raises(AirflowException) as ctx:
            operator.execute(None)
        assert str(ctx.value) == 'Recursive glob patterns using `**` are not supported'

    @mock.patch('airflow.providers.microsoft.azure.transfers.local_to_adls.AzureDataLakeHook')
    def test_extra_options_is_passed(self, mock_hook):
        if False:
            print('Hello World!')
        operator = LocalFilesystemToADLSOperator(task_id=TASK_ID, local_path=LOCAL_PATH, remote_path=REMOTE_PATH, extra_upload_options={'run': False})
        operator.execute(None)
        mock_hook.return_value.upload_file.assert_called_once_with(local_path=LOCAL_PATH, remote_path=REMOTE_PATH, nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304, run=False)