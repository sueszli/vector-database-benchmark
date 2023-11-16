import os
import job_search_delete_tenant
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_delete_tenant(capsys, tenant):
    if False:
        for i in range(10):
            print('nop')
    job_search_delete_tenant.delete_tenant(PROJECT_ID, tenant)
    (out, _) = capsys.readouterr()
    assert 'Deleted' in out