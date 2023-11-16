import os
from unittest import mock
from documentai.snippets import deploy_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aaaaaaaaa'
processor_version_id = 'xxxxxxxxxx'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.deploy_processor_version')
@mock.patch('google.api_core.operation.Operation')
def test_deploy_processor_version(operation_mock, deploy_processor_version_mock, capsys):
    if False:
        for i in range(10):
            print('nop')
    deploy_processor_version_mock.return_value = operation_mock
    deploy_processor_version_sample.deploy_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id)
    deploy_processor_version_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'operation' in out