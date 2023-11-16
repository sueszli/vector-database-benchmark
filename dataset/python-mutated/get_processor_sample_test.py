import os
from documentai.snippets import get_processor_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '52a38e080c1a7296'

def test_get_processor(capsys):
    if False:
        print('Hello World!')
    get_processor_sample.get_processor_sample(project_id=project_id, location=location, processor_id=processor_id)
    (out, _) = capsys.readouterr()
    assert 'Processor Name:' in out
    assert 'Processor Display Name:' in out
    assert 'OCR_PROCESSOR' in out
    assert processor_id in out