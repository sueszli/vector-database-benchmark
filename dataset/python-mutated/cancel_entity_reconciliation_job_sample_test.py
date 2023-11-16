import os
from google.api_core.exceptions import InvalidArgument, NotFound
import cancel_entity_reconciliation_job_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
job_id = '5285051433452986163'

def test_cancel_entity_reconciliation_job(capsys):
    if False:
        return 10
    try:
        cancel_entity_reconciliation_job_sample.cancel_entity_reconciliation_job_sample(project_id=project_id, location=location, job_id=job_id)
    except (InvalidArgument, NotFound) as e:
        print(e.message)
    (out, _) = capsys.readouterr()
    assert 'cancel' in out