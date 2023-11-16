import os
from typing import Iterator
import uuid
import create_trigger as ct
import delete_trigger as dt
import google.api_core.exceptions
import google.cloud.storage
import list_triggers as lt
import pytest
import update_trigger as ut
UNIQUE_STRING = str(uuid.uuid4()).split('-')[0]
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
TEST_BUCKET_NAME = GCLOUD_PROJECT + '-dlp-python-client-test' + UNIQUE_STRING
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')
RESOURCE_FILE_NAMES = ['test.txt', 'test.png', 'harmless.txt', 'accounts.txt']
TEST_TRIGGER_ID = 'test-trigger' + UNIQUE_STRING

@pytest.fixture(scope='module')
def bucket() -> Iterator[google.cloud.storage.bucket.Bucket]:
    if False:
        for i in range(10):
            print('nop')
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

def test_create_list_update_and_delete_trigger(bucket: google.cloud.storage.bucket.Bucket, capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    try:
        ct.create_trigger(GCLOUD_PROJECT, bucket.name, 7, ['FIRST_NAME', 'EMAIL_ADDRESS', 'PHONE_NUMBER'], trigger_id=TEST_TRIGGER_ID)
    except google.api_core.exceptions.InvalidArgument:
        dt.delete_trigger(GCLOUD_PROJECT, TEST_TRIGGER_ID)
        (out, _) = capsys.readouterr()
        assert TEST_TRIGGER_ID in out
        ct.create_trigger(GCLOUD_PROJECT, bucket.name, 7, ['FIRST_NAME', 'EMAIL_ADDRESS', 'PHONE_NUMBER'], trigger_id=TEST_TRIGGER_ID, auto_populate_timespan=True)
    (out, _) = capsys.readouterr()
    assert TEST_TRIGGER_ID in out
    lt.list_triggers(GCLOUD_PROJECT)
    (out, _) = capsys.readouterr()
    assert TEST_TRIGGER_ID in out
    ut.update_trigger(GCLOUD_PROJECT, ['US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER'], TEST_TRIGGER_ID)
    (out, _) = capsys.readouterr()
    assert TEST_TRIGGER_ID in out
    assert 'US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER' in out
    dt.delete_trigger(GCLOUD_PROJECT, TEST_TRIGGER_ID)
    (out, _) = capsys.readouterr()
    assert TEST_TRIGGER_ID in out