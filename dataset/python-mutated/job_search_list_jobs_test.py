import os
import job_search_list_jobs
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_jobs(capsys, tenant, company):
    if False:
        return 10
    filter = f'companyName="projects/{PROJECT_ID}/companies/{company}"'
    jobs = job_search_list_jobs.list_jobs(PROJECT_ID, tenant, filter)
    for job in jobs:
        assert 'projects/' in job