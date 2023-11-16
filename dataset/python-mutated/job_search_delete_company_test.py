import os
import job_search_delete_company
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_delete_company(capsys, tenant, company):
    if False:
        return 10
    (out, _) = capsys.readouterr()
    job_search_delete_company.delete_company(PROJECT_ID, tenant, company)
    (out, _) = capsys.readouterr()
    assert 'Deleted' in out