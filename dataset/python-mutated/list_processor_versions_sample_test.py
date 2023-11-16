import os
from documentai.snippets import list_processor_versions_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '52a38e080c1a7296'

def test_list_processor_versions(capsys):
    if False:
        while True:
            i = 10
    list_processor_versions_sample.list_processor_versions_sample(project_id=project_id, location=location, processor_id=processor_id)
    (out, _) = capsys.readouterr()
    assert 'Processor Version: pretrained-ocr' in out
    assert 'Display Name: Google Stable' in out
    assert 'Display Name: Google Release Candidate' in out
    assert 'DEPLOYED' in out