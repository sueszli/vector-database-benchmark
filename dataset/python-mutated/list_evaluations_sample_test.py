import os
from documentai.snippets import list_evaluations_sample
location = 'us'
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
processor_id = 'feacd98c28866ede'
processor_version_id = 'stable'

def test_list_evaluations(capsys):
    if False:
        print('Hello World!')
    list_evaluations_sample.list_evaluations_sample(project_id=project_id, location=location, processor_id=processor_id, processor_version_id=processor_version_id)
    (out, _) = capsys.readouterr()
    assert 'Evaluation' in out