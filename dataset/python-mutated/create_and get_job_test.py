import os
from typing import Iterator
import uuid
import create_job
import delete_job
import get_job
import google.cloud.storage
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
UNIQUE_STRING = str(uuid.uuid4()).split('-')[0]
TEST_BUCKET_NAME = GCLOUD_PROJECT + '-dlp-python-client-test' + UNIQUE_STRING
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')
RESOURCE_FILE_NAMES = ['test.txt', 'test.png', 'harmless.txt', 'accounts.txt']
test_job_id = f'test-job-{uuid.uuid4()}'

@pytest.fixture(scope='module')
def bucket() -> Iterator[google.cloud.storage.bucket.Bucket]:
    if False:
        print('Hello World!')
    client = google.cloud.storage.Client()
    try:
        bucket = client.get_bucket(TEST_BUCKET_NAME)
    except google.cloud.exceptions.NotFound:
        bucket = client.create_bucket(TEST_BUCKET_NAME)
    blobs = []
    for name in RESOURCE_FILE_NAMES:
        path = os.path.join(RESOURCE_DIRECTORY, name)
        blob = bucket.blob(name)
        blob.upload_from_filename(path)
        blobs.append(blob)
    yield bucket
    for blob in blobs:
        try:
            blob.delete()
        except google.cloud.exceptions.NotFound:
            print('Issue during teardown, missing blob')
    bucket.delete()

def test_create_dlp_job(bucket: google.cloud.storage.bucket.Bucket, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    create_job.create_dlp_job(GCLOUD_PROJECT, bucket.name, ['EMAIL_ADDRESS', 'CREDIT_CARD_NUMBER'], job_id=test_job_id)
    (out, _) = capsys.readouterr()
    assert test_job_id in out
    job_name = f'i-{test_job_id}'
    get_job.get_dlp_job(GCLOUD_PROJECT, job_name)
    (out, _) = capsys.readouterr()
    assert job_name in out
    delete_job.delete_dlp_job(GCLOUD_PROJECT, job_name)