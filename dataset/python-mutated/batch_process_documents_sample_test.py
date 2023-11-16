import os
from uuid import uuid4
from documentai.snippets import batch_process_documents_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '90484cfdedb024f6'
processor_version_id = 'pretrained-form-parser-v1.0-2020-09-23'
gcs_input_uri = 'gs://cloud-samples-data/documentai/invoice.pdf'
gcs_input_prefix = 'gs://cloud-samples-data/documentai/workflows/'
input_mime_type = 'application/pdf'
gcs_output_uri = f'gs://document-ai-python/{uuid4()}/'
field_mask = 'text,pages.pageNumber'

def test_batch_process_documents(capsys):
    if False:
        return 10
    batch_process_documents_sample.batch_process_documents(project_id=project_id, location=location, processor_id=processor_id, gcs_input_uri=gcs_input_uri, input_mime_type=input_mime_type, gcs_output_uri=gcs_output_uri, field_mask=field_mask)
    (out, _) = capsys.readouterr()
    assert 'operation' in out
    assert 'Fetching' in out
    assert 'text:' in out

def test_batch_process_documents_processor_version(capsys):
    if False:
        print('Hello World!')
    batch_process_documents_sample.batch_process_documents(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id, gcs_input_uri=gcs_input_uri, input_mime_type=input_mime_type, gcs_output_uri=gcs_output_uri, field_mask=field_mask)
    (out, _) = capsys.readouterr()
    assert 'operation' in out
    assert 'Fetching' in out
    assert 'text:' in out

def test_batch_process_documents_gcs_prefix(capsys):
    if False:
        return 10
    batch_process_documents_sample.batch_process_documents(project_id=project_id, location=location, processor_id=processor_id, gcs_input_prefix=gcs_input_prefix, gcs_output_uri=gcs_output_uri, field_mask=field_mask)
    (out, _) = capsys.readouterr()
    assert 'operation' in out
    assert 'Fetching' in out
    assert 'text:' in out