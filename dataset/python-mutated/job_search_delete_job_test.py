import os
import job_search_delete_job
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_delete_job(capsys, tenant, job):
    if False:
        return 10
    job_search_delete_job.delete_job(PROJECT_ID, tenant, job)
    (out, _) = capsys.readouterr()
    assert 'Deleted' in out