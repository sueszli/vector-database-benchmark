import json
import backoff
from google.api_core.exceptions import RetryError
from googleapiclient.errors import HttpError
import pytest
import create_client
import create_client_apiary

@pytest.fixture()
def job_filter(project_id: str):
    if False:
        i = 10
        return i + 15
    yield json.dumps({'project_id': project_id})

@backoff.on_exception(backoff.expo, (RetryError,), max_time=60)
def test_create_client(job_filter: str):
    if False:
        i = 10
        return i + 15
    client = create_client.create_transfer_client()
    client.list_transfer_jobs({'filter': job_filter, 'page_size': 1})

@backoff.on_exception(backoff.expo, (HttpError,), max_time=60)
def test_create_client_apiary(job_filter: str):
    if False:
        for i in range(10):
            print('nop')
    client = create_client_apiary.create_transfer_client()
    client.transferJobs().list(filter=job_filter, pageSize=1).execute()