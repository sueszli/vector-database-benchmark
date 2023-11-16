import sys
import uuid
from google.cloud import storage
import pytest
from ..batch_write_storage import write_to_cloud_storage
bucket_name = f'test-bucket-{uuid.uuid4()}'
storage_client = storage.Client()

@pytest.fixture(scope='function')
def setup_and_teardown():
    if False:
        print('Hello World!')
    try:
        bucket = storage_client.create_bucket(bucket_name)
        yield
    finally:
        bucket.delete(force=True)

def test_write_to_cloud_storage(setup_and_teardown):
    if False:
        i = 10
        return i + 15
    sys.argv = ['', f'--output=gs://{bucket_name}/output/out-']
    write_to_cloud_storage()
    blobs = list(storage_client.list_blobs(bucket_name))
    assert blobs
    for blob in blobs:
        assert blob.name.endswith('.txt')