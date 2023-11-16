import os
from typing import Iterator
import uuid
import delete_job as jobs
import google.cloud.dlp
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_COLUMN_NAME = 'zip_code'
TEST_TABLE_PROJECT_ID = 'bigquery-public-data'
TEST_DATASET_ID = 'san_francisco'
TEST_TABLE_ID = 'bikeshare_trips'
test_job_id = f'test-job-{uuid.uuid4()}'

@pytest.fixture(scope='module')
def test_job_name() -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{GCLOUD_PROJECT}'
    risk_job = {'privacy_metric': {'categorical_stats_config': {'field': {'name': TEST_COLUMN_NAME}}}, 'source_table': {'project_id': TEST_TABLE_PROJECT_ID, 'dataset_id': TEST_DATASET_ID, 'table_id': TEST_TABLE_ID}}
    response = dlp.create_dlp_job(request={'parent': parent, 'risk_job': risk_job, 'job_id': test_job_id})
    full_path = response.name
    job_name = full_path[full_path.rfind('/') + 1:]
    yield job_name
    try:
        dlp.delete_dlp_job(request={'name': full_path})
    except google.api_core.exceptions.NotFound:
        print('Issue during teardown, missing job')

def test_delete_dlp_job(test_job_name: str) -> None:
    if False:
        while True:
            i = 10
    jobs.delete_dlp_job(GCLOUD_PROJECT, test_job_name)