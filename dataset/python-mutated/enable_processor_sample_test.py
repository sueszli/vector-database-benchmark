import os
from documentai.snippets import disable_processor_sample
from documentai.snippets import enable_processor_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = '351535be16606fe3'

def test_enable_processor(capsys):
    if False:
        print('Hello World!')
    enable_processor_sample.enable_processor_sample(project_id=project_id, location=location, processor_id=processor_id)
    (out, _) = capsys.readouterr()
    assert 'projects' in out or 'Processor' in out
    disable_processor_sample.disable_processor_sample(project_id=project_id, location=location, processor_id=processor_id)