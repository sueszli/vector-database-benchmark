import os
import job_search_get_tenant
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_get_tenant(capsys, tenant):
    if False:
        while True:
            i = 10
    job_search_get_tenant.get_tenant(PROJECT_ID, tenant)
    (out, _) = capsys.readouterr()
    assert 'Name: ' in out