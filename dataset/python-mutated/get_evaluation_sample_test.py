import os
from unittest import mock
from documentai.snippets import get_evaluation_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'a35310a144a6e4f8'
processor_version_id = '2af620b2fd4d1fcf'
evaluation_id = '55cdab6206095055'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.get_evaluation')
@mock.patch('google.cloud.documentai.Evaluation')
def test_get_evaluation(evaluation_mock, get_evaluation_mock, capsys):
    if False:
        i = 10
        return i + 15
    get_evaluation_mock.return_value = evaluation_mock
    get_evaluation_sample.get_evaluation_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id, evaluation_id=evaluation_id)
    get_evaluation_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'Create Time' in out