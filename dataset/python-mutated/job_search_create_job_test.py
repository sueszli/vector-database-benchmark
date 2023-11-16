import os
import uuid
import pytest
import job_search_create_job
import job_search_delete_job
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
JOB_EXT_UNIQUE_ID = f'TEST_JOB_{uuid.uuid4()}'

def test_create_job(capsys, tenant, company, cleaner):
    if False:
        print('Hello World!')
    job_name = job_search_create_job.create_job(PROJECT_ID, tenant, company, JOB_EXT_UNIQUE_ID, 'www.example.com')
    (out, _) = capsys.readouterr()
    assert 'Created job:' in out
    job_id = job_name.split('/')[-1]
    cleaner.append(job_id)

@pytest.fixture(scope='module')
def cleaner(tenant):
    if False:
        for i in range(10):
            print('nop')
    jobs = []
    yield jobs
    for job_id in jobs:
        job_search_delete_job.delete_job(PROJECT_ID, tenant, job_id)