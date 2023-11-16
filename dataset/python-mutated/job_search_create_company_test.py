import os
import uuid
import pytest
import job_search_create_company
import job_search_delete_company
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
COMPANY_EXT_ID = f'COMPANY_EXT_ID_{uuid.uuid4()}'

def test_create_company(capsys, tenant, cleaner):
    if False:
        while True:
            i = 10
    company_name = job_search_create_company.create_company(PROJECT_ID, tenant, 'Test Company Name', COMPANY_EXT_ID)
    (out, _) = capsys.readouterr()
    assert 'Created' in out
    assert 'Name:' in out
    company_id = company_name.split('/')[-1]
    cleaner.append(company_id)

@pytest.fixture(scope='module')
def cleaner(tenant):
    if False:
        for i in range(10):
            print('nop')
    companies = []
    yield companies
    for company_id in companies:
        job_search_delete_company.delete_company(PROJECT_ID, tenant, company_id)