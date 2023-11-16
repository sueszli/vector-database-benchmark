import os
import job_search_get_company
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_job_search_get_company(capsys, tenant, company):
    if False:
        for i in range(10):
            print('nop')
    job_search_get_company.get_company(PROJECT_ID, tenant, company)
    (out, _) = capsys.readouterr()
    assert 'Company name:' in out