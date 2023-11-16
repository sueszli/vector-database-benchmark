import os
import job_search_get_job
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_job_search_get_job(capsys, tenant, job):
    if False:
        print('Hello World!')
    job_search_get_job.get_job(PROJECT_ID, tenant, job)
    (out, _) = capsys.readouterr()
    assert 'Job name:' in out