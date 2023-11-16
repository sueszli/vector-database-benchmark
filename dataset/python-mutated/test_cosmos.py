from __future__ import annotations
from unittest import mock
from airflow.providers.microsoft.azure.sensors.cosmos import AzureCosmosDocumentSensor
DB_NAME = 'test-db-name'
COLLECTION_NAME = 'test-db-collection-name'
DOCUMENT_ID = 'test-document-id'

class TestAzureCosmosSensor:

    @mock.patch('airflow.providers.microsoft.azure.sensors.cosmos.AzureCosmosDBHook')
    def test_should_call_hook_with_args(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_instance = mock_hook.return_value
        mock_instance.get_document.return_value = True
        sensor = AzureCosmosDocumentSensor(task_id='test-task-1', database_name=DB_NAME, collection_name=COLLECTION_NAME, document_id=DOCUMENT_ID)
        result = sensor.poke(None)
        mock_instance.get_document.assert_called_once_with(DOCUMENT_ID, DB_NAME, COLLECTION_NAME)
        assert result is True

    @mock.patch('airflow.providers.microsoft.azure.sensors.cosmos.AzureCosmosDBHook')
    def test_should_return_false_on_no_document(self, mock_hook):
        if False:
            i = 10
            return i + 15
        mock_instance = mock_hook.return_value
        mock_instance.get_document.return_value = None
        sensor = AzureCosmosDocumentSensor(task_id='test-task-2', database_name=DB_NAME, collection_name=COLLECTION_NAME, document_id=DOCUMENT_ID)
        result = sensor.poke(None)
        mock_instance.get_document.assert_called_once_with(DOCUMENT_ID, DB_NAME, COLLECTION_NAME)
        assert result is False