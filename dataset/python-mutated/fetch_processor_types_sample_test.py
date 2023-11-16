import os
from documentai.snippets import fetch_processor_types_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']

def test_fetch_processor_types(capsys):
    if False:
        print('Hello World!')
    fetch_processor_types_sample.fetch_processor_types_sample(project_id=project_id, location=location)
    (out, _) = capsys.readouterr()
    assert 'OCR_PROCESSOR' in out
    assert 'FORM_PARSER_PROCESSOR' in out