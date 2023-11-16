import os
from documentai.snippets import set_default_processor_version_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'aeb8cea219b7c272'
current_default_processor_version = 'pretrained-ocr-v1.0-2020-09-23'
new_default_processor_version = 'pretrained-ocr-v1.1-2022-09-12'

def test_set_default_processor_version(capsys):
    if False:
        for i in range(10):
            print('nop')
    set_default_processor_version_sample.set_default_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=new_default_processor_version)
    (out, _) = capsys.readouterr()
    assert 'operation' in out
    set_default_processor_version_sample.set_default_processor_version_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=current_default_processor_version)