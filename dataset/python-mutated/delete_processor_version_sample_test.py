import os
from unittest import mock
from documentai.snippets import delete_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aaaaaaaaa'
processor_version_id = 'xxxxxxxxxx'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.delete_processor_version')
@mock.patch('google.api_core.operation.Operation')
def test_delete_processor_version(operation_mock, delete_processor_version_mock, capsys):
    if False:
        for i in range(10):
            print('nop')
    delete_processor_version_mock.return_value = operation_mock
    delete_processor_version_sample.delete_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id)
    delete_processor_version_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'operation' in out