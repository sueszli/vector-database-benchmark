import os
from unittest import mock
from documentai.snippets import evaluate_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aaaaaaaaa'
processor_version_id = 'xxxxxxxxxx'
gcs_input_uri = 'gs://bucket/directory/'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.evaluate_processor_version')
@mock.patch('google.cloud.documentai.EvaluateProcessorVersionResponse')
@mock.patch('google.api_core.operation.Operation')
def test_evaluate_processor_version(operation_mock, evaluate_processor_version_response_mock, evaluate_processor_version_mock, capsys):
    if False:
        while True:
            i = 10
    operation_mock.result.return_value = evaluate_processor_version_response_mock
    evaluate_processor_version_mock.return_value = operation_mock
    evaluate_processor_version_sample.evaluate_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id, gcs_input_uri=gcs_input_uri)
    evaluate_processor_version_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'operation' in out