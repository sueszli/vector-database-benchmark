import os
import uuid
import pytest
import job_search_create_job_custom_attributes
import job_search_delete_job
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
JOB_EXT_UNIQUE_ID = f'TEST_JOB_{uuid.uuid4()}'

def test_create_job_with_attributes(capsys, tenant, company, cleaner):
    if False:
        return 10
    job_name = job_search_create_job_custom_attributes.create_job(PROJECT_ID, tenant, company, JOB_EXT_UNIQUE_ID)
    (out, _) = capsys.readouterr()
    assert 'Created job:' in out
    job_id = job_name.split('/')[-1]
    cleaner.append(job_id)

@pytest.fixture(scope='module')
def cleaner(tenant):
    if False:
        while True:
            i = 10
    jobs = []
    yield jobs
    for job_id in jobs:
        job_search_delete_job.delete_job(PROJECT_ID, tenant, job_id)