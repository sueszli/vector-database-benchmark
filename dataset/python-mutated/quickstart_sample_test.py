import os
from uuid import uuid4
from documentai.snippets import quickstart_sample
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_display_name = f'test-processor-{uuid4()}'
file_path = 'resources/invoice.pdf'

def test_quickstart(capsys):
    if False:
        return 10
    processor = quickstart_sample.quickstart(project_id=project_id, location=location, processor_display_name=processor_display_name, file_path=file_path)
    (out, _) = capsys.readouterr()
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com'))
    operation = client.delete_processor(name=processor.name)
    operation.result()
    assert 'Processor Name:' in out
    assert 'text:' in out
    assert 'Invoice' in out