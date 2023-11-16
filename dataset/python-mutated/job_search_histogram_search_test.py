import os
import job_search_histogram_search
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_search_jobs_histogram(tenant):
    if False:
        print('Hello World!')
    query = 'count(base_compensation, [bucket(12, 20)])'
    jobs = job_search_histogram_search.search_jobs(PROJECT_ID, tenant, query)
    for job in jobs:
        assert 'projects/' in job