import os
from unittest import mock
from uuid import uuid4
from documentai.snippets import create_processor_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_display_name = f'test-processor-{uuid4()}'
processor_type = 'OCR_PROCESSOR'

@mock.patch('google.cloud.documentai.DocumentProcessorServiceClient.create_processor')
@mock.patch('google.cloud.documentai.Processor')
def test_create_processor(create_processor_mock, processor_mock, capsys):
    if False:
        i = 10
        return i + 15
    create_processor_mock.return_value = processor_mock
    create_processor_sample.create_processor_sample(project_id=project_id, location=location, processor_display_name=processor_display_name, processor_type=processor_type)
    create_processor_mock.assert_called_once()
    (out, _) = capsys.readouterr()
    assert 'Processor Name:' in out
    assert 'Processor Display Name:' in out
    assert 'Processor Type:' in out