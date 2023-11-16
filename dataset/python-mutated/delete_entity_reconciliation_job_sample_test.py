import os
from google.api_core.exceptions import InvalidArgument, NotFound
import delete_entity_reconciliation_job_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
job_id = '5285051433452986164'

def test_delete_entity_reconciliation_job(capsys):
    if False:
        print('Hello World!')
    try:
        delete_entity_reconciliation_job_sample.delete_entity_reconciliation_job_sample(project_id=project_id, location=location, job_id=job_id)
    except (InvalidArgument, NotFound) as e:
        print(e.message)
    (out, _) = capsys.readouterr()
    assert 'projects/' in out