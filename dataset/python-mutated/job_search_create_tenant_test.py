import os
import uuid
import pytest
import job_search_create_tenant
import job_search_delete_tenant
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
TENANT_EXT_UNIQUE_ID = f'TEST_TENANT_{uuid.uuid4()}'

def test_create_tenant(capsys, cleaner):
    if False:
        for i in range(10):
            print('nop')
    tenant_name = job_search_create_tenant.create_tenant(PROJECT_ID, TENANT_EXT_UNIQUE_ID)
    (out, _) = capsys.readouterr()
    assert 'Created Tenant' in out
    assert 'Name:' in out
    tenant_id = tenant_name.split('/')[-1]
    cleaner.append(tenant_id)

@pytest.fixture(scope='module')
def cleaner():
    if False:
        while True:
            i = 10
    tenants = []
    yield tenants
    for tenant_id in tenants:
        job_search_delete_tenant.delete_tenant(PROJECT_ID, tenant_id)