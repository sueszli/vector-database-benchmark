from __future__ import annotations
from unittest import mock
from airflow.providers.microsoft.azure.operators.adls import ADLSDeleteOperator
TASK_ID = 'test-adls-list-operator'
TEST_PATH = 'test'

class TestAzureDataLakeStorageDeleteOperator:

    @mock.patch('airflow.providers.microsoft.azure.operators.adls.AzureDataLakeHook')
    def test_execute(self, mock_hook):
        if False:
            print('Hello World!')
        operator = ADLSDeleteOperator(task_id=TASK_ID, path=TEST_PATH)
        operator.execute(None)
        mock_hook.return_value.remove.assert_called_once_with(path=TEST_PATH, recursive=False, ignore_not_found=True)