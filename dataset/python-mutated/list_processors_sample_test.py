import os
from documentai.snippets import list_processors_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_processors(capsys):
    if False:
        return 10
    list_processors_sample.list_processors_sample(project_id=project_id, location=location)
    (out, _) = capsys.readouterr()
    assert 'Processor Name:' in out
    assert 'Processor Display Name:' in out
    assert 'OCR_PROCESSOR' in out
    assert 'FORM_PARSER_PROCESSOR' in out