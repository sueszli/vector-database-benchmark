import os
import list_entity_reconciliation_jobs_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'

def test_list_entity_reconciliation_jobs(capsys):
    if False:
        i = 10
        return i + 15
    list_entity_reconciliation_jobs_sample.list_entity_reconciliation_jobs_sample(project_id=project_id, location=location)
    (out, _) = capsys.readouterr()
    assert 'Job: projects/' in out
    assert 'Input Table: projects/' in out
    assert 'Output Dataset: projects/' in out
    assert 'State:' in out