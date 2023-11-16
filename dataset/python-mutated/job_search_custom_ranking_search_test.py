import os
import job_search_custom_ranking_search
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_search_jobs_custom_ranking(tenant):
    if False:
        while True:
            i = 10
    jobs = job_search_custom_ranking_search.search_jobs(PROJECT_ID, tenant)
    for job in jobs:
        assert 'projects/' in job