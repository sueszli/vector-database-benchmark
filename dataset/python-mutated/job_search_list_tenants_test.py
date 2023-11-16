import os
import uuid
from google.cloud import talent
import pytest
import job_search_list_tenants
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture(scope='module')
def test_tenant():
    if False:
        for i in range(10):
            print('nop')
    client = talent.TenantServiceClient()
    external_id = f'test_tenant_{uuid.uuid4().hex}'
    parent = f'projects/{PROJECT_ID}'
    tenant = {'external_id': external_id}
    resp = client.create_tenant(parent=parent, tenant=tenant)
    yield resp
    client.delete_tenant(name=resp.name)

def test_list_tenants(capsys, test_tenant):
    if False:
        while True:
            i = 10
    job_search_list_tenants.list_tenants(PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Tenant Name:' in out