import os
from typing import Tuple
import uuid
import backoff
from botocore.exceptions import ClientError
from google.cloud import storage
import pytest
import list_gcs_buckets
import list_gcs_objects
PROJECT_ID = os.environ['MAIN_GOOGLE_CLOUD_PROJECT']
SERVICE_ACCOUNT_EMAIL = os.environ['HMAC_KEY_TEST_SERVICE_ACCOUNT']
STORAGE_CLIENT = storage.Client(project=PROJECT_ID)

@pytest.fixture(scope='module')
def hmac_fixture() -> Tuple[storage.hmac_key.HMACKeyMetadata, str]:
    if False:
        while True:
            i = 10
    '\n    Creates an HMAC Key and secret to supply to the S3 SDK tests. The key\n    will be deleted after the test session.\n    '
    (hmac_key, secret) = STORAGE_CLIENT.create_hmac_key(service_account_email=SERVICE_ACCOUNT_EMAIL, project_id=PROJECT_ID)
    yield (hmac_key, secret)
    hmac_key.state = 'INACTIVE'
    hmac_key.update()
    hmac_key.delete()

@pytest.fixture(scope='module')
def test_bucket() -> storage.Bucket:
    if False:
        while True:
            i = 10
    'Yields a bucket that is deleted after the test completes.'
    bucket = None
    while bucket is None or bucket.exists():
        bucket_name = f'bucket-storage-s3-test-{uuid.uuid4()}'
        bucket = storage.Client().bucket(bucket_name)
    bucket.create()
    yield bucket
    bucket.delete(force=True)

@pytest.fixture(scope='module')
def test_blob(test_bucket: storage.Bucket) -> storage.Blob:
    if False:
        i = 10
        return i + 15
    'Yields a blob that is deleted after the test completes.'
    bucket = test_bucket
    blob = bucket.blob(f'storage_snippets_test_sigil-{uuid.uuid4()}')
    blob.upload_from_string("Hello, is it me you're looking for?")
    yield blob

@backoff.on_exception(backoff.constant, ClientError, interval=1, max_time=15)
def test_list_buckets(hmac_fixture: Tuple[storage.hmac_key.HMACKeyMetadata, str], test_bucket: storage.Bucket) -> None:
    if False:
        while True:
            i = 10
    result = list_gcs_buckets.list_gcs_buckets(google_access_key_id=hmac_fixture[0].access_id, google_access_key_secret=hmac_fixture[1])
    assert test_bucket.name in result

@backoff.on_exception(backoff.constant, ClientError, interval=1, max_time=15)
def test_list_blobs(hmac_fixture: Tuple[storage.hmac_key.HMACKeyMetadata, str], test_bucket: storage.Bucket, test_blob: storage.Blob) -> None:
    if False:
        i = 10
        return i + 15
    result = list_gcs_objects.list_gcs_objects(google_access_key_id=hmac_fixture[0].access_id, google_access_key_secret=hmac_fixture[1], bucket_name=test_bucket.name)
    assert test_blob.name in result