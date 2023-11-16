import os
import job_search_list_companies
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_companies(tenant):
    if False:
        while True:
            i = 10
    results = job_search_list_companies.list_companies(PROJECT_ID, tenant)
    for company in results:
        assert 'projects/' in company.name