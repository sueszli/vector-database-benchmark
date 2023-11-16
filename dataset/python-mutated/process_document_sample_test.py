import os
from documentai.snippets import process_document_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '90484cfdedb024f6'
processor_version_id = 'stable'
file_path = 'resources/invoice.pdf'
mime_type = 'application/pdf'
field_mask = 'text,pages.pageNumber'

def test_process_document(capsys):
    if False:
        i = 10
        return i + 15
    process_document_sample.process_document_sample(project_id=project_id, location=location, processor_id=processor_id, file_path=file_path, mime_type=mime_type, field_mask=field_mask)
    (out, _) = capsys.readouterr()
    assert 'text:' in out
    assert 'Invoice' in out

def test_process_document_processor_version(capsys):
    if False:
        return 10
    process_document_sample.process_document_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id, file_path=file_path, mime_type=mime_type, field_mask=field_mask)
    (out, _) = capsys.readouterr()
    assert 'text:' in out
    assert 'Invoice' in out