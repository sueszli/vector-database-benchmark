import os
import uuid
import backoff
from google.api_core.exceptions import InvalidArgument
from google.cloud import storage
import pytest
import quickstart_batchgetassetshistory
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = f'assets-{uuid.uuid4().hex}'

@pytest.fixture(scope='module')
def storage_client():
    if False:
        i = 10
        return i + 15
    yield storage.Client(project=PROJECT)

@pytest.fixture(scope='module')
def asset_bucket(storage_client):
    if False:
        for i in range(10):
            print('nop')
    bucket = storage_client.create_bucket(BUCKET, project=PROJECT)
    yield BUCKET
    try:
        bucket.delete(force=True)
    except Exception as e:
        print(f'Failed to delete bucket{BUCKET}')
        raise e

def test_batch_get_assets_history(asset_bucket, capsys):
    if False:
        print('Hello World!')
    bucket_asset_name = f'//storage.googleapis.com/{BUCKET}'
    asset_names = [bucket_asset_name]

    @backoff.on_exception(backoff.expo, (AssertionError, InvalidArgument), max_time=60)
    def eventually_consistent_test():
        if False:
            while True:
                i = 10
        quickstart_batchgetassetshistory.batch_get_assets_history(PROJECT, asset_names)
        (out, _) = capsys.readouterr()
        assert bucket_asset_name in out
    eventually_consistent_test()