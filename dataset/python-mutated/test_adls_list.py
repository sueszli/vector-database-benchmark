from __future__ import annotations
from unittest import mock
from airflow.providers.microsoft.azure.operators.adls import ADLSListOperator
TASK_ID = 'test-adls-list-operator'
TEST_PATH = 'test/*'
MOCK_FILES = ['test/TEST1.csv', 'test/TEST2.csv', 'test/path/TEST3.csv', 'test/path/PARQUET.parquet', 'test/path/PIC.png']

class TestAzureDataLakeStorageListOperator:

    @mock.patch('airflow.providers.microsoft.azure.operators.adls.AzureDataLakeHook')
    def test_execute(self, mock_hook):
        if False:
            return 10
        mock_hook.return_value.list.return_value = MOCK_FILES
        operator = ADLSListOperator(task_id=TASK_ID, path=TEST_PATH)
        files = operator.execute(None)
        mock_hook.return_value.list.assert_called_once_with(path=TEST_PATH)
        assert sorted(files) == sorted(MOCK_FILES)