import os
from unittest import mock
from documentai.snippets import train_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aaaaaaaaa'
processor_version_display_name = 'new-processor-version'
train_data_uri = 'gs://bucket/directory/'
test_data_uri = 'gs://bucket/directory/'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.train_processor_version')
@mock.patch('google.cloud.documentai.TrainProcessorVersionResponse')
@mock.patch('google.cloud.documentai.TrainProcessorVersionMetadata')
@mock.patch('google.api_core.operation.Operation')
def test_train_processor_version(operation_mock, train_processor_version_metadata_mock, train_processor_version_response_mock, train_processor_version_mock, capsys):
    if False:
        return 10
    operation_mock.result.return_value = train_processor_version_response_mock
    operation_mock.metadata.return_value = train_processor_version_metadata_mock
    train_processor_version_mock.return_value = operation_mock
    train_processor_version_sample.train_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_display_name=processor_version_display_name, train_data_uri=train_data_uri, test_data_uri=test_data_uri)
    train_processor_version_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'operation' in out