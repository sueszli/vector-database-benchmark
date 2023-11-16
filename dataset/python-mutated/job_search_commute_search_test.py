from datetime import date
import os
import pytest
import job_search_commute_search
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.mark.skipif(date.today() < date(2023, 4, 25), reason='Addressed by product team until this date, b/277494438')
def test_commute_search(tenant):
    if False:
        for i in range(10):
            print('nop')
    jobs = job_search_commute_search.search_jobs(PROJECT_ID, tenant)
    for job in jobs:
        assert 'projects/' in job