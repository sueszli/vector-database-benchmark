import uuid
from flaky import flaky
import google.auth
from google.cloud import batch_v1
from google.cloud import storage
import pytest
from .test_basics import _test_body
from ..create.create_with_mounted_bucket import create_script_job_with_bucket
PROJECT = google.auth.default()[1]
REGION = 'europe-north1'
TIMEOUT = 600
WAIT_STATES = {batch_v1.JobStatus.State.STATE_UNSPECIFIED, batch_v1.JobStatus.State.QUEUED, batch_v1.JobStatus.State.RUNNING, batch_v1.JobStatus.State.SCHEDULED}

@pytest.fixture
def job_name():
    if False:
        return 10
    return f'test-job-{uuid.uuid4().hex[:10]}'

@pytest.fixture()
def test_bucket():
    if False:
        while True:
            i = 10
    bucket_name = f'test-bucket-{uuid.uuid4().hex[:8]}'
    client = storage.Client()
    client.create_bucket(bucket_name, location='eu')
    yield bucket_name
    bucket = client.get_bucket(bucket_name)
    bucket.delete(force=True)

def _test_bucket_content(test_bucket):
    if False:
        return 10
    client = storage.Client()
    bucket = client.get_bucket(test_bucket)
    file_name_template = 'output_task_{task_number}.txt'
    file_content_template = 'Hello world from task {task_number}.\n'
    for i in range(4):
        blob = bucket.blob(file_name_template.format(task_number=i))
        content = blob.download_as_bytes().decode()
        assert content == file_content_template.format(task_number=i)

@flaky(max_runs=3, min_passes=1)
def test_bucket_job(job_name, test_bucket):
    if False:
        while True:
            i = 10
    job = create_script_job_with_bucket(PROJECT, REGION, job_name, test_bucket)
    _test_body(job, lambda : _test_bucket_content(test_bucket))