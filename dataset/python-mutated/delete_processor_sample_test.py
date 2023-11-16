import os
from unittest import mock
from documentai.snippets import delete_processor_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aaaaaaaaa'
parent = f'projects/{project_id}/locations/{location}/processors/{processor_id}'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.delete_processor')
@mock.patch('google.api_core.operation.Operation')
def test_delete_processor(operation_mock, delete_processor_mock, capsys):
    if False:
        print('Hello World!')
    delete_processor_mock.return_value = operation_mock
    delete_processor_sample.delete_processor_sample(project_id=project_id, location=location, processor_id=processor_id)
    delete_processor_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'operation' in out