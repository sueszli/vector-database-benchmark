import os
from documentai.snippets import get_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '52a38e080c1a7296'
processor_version_id = 'pretrained-ocr-v1.0-2020-09-23'

def test_get_processor_version(capsys):
    if False:
        return 10
    get_processor_version_sample.get_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id)
    (out, _) = capsys.readouterr()
    assert 'Processor Version: pretrained-ocr' in out
    assert 'Display Name: Google Stable' in out
    assert 'DEPLOYED' in out