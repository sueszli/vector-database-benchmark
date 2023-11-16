import os
from documentai.snippets import review_document_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'b7054d67d76c39f1'
file_path = 'resources/invoice.pdf'
mime_type = 'application/pdf'

def test_review_document(capsys):
    if False:
        print('Hello World!')
    review_document_sample.review_document_sample(project_id=project_id, location=location, processor_id=processor_id, file_path=file_path, mime_type=mime_type)
    (out, _) = capsys.readouterr()
    assert 'projects/' in out
    assert 'locations/' in out
    assert 'operations/' in out