import os
import time
from typing import Iterator
import uuid
import create_stored_infotype as create_si
import google.api_core.exceptions
import google.cloud.dlp_v2
import google.cloud.exceptions
import google.cloud.storage
import inspect_with_stored_infotype as inspect_si
import pytest
import update_stored_infotype as update_si
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
UNIQUE_STRING = str(uuid.uuid4()).split('-')[0]
TEST_BUCKET_NAME = GCLOUD_PROJECT + '-dlp-python-client-test' + UNIQUE_STRING
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')
RESOURCE_FILE_NAMES = ['term_list.txt']
STORED_INFO_TYPE_ID = 'github-user-names' + UNIQUE_STRING
DLP_CLIENT = google.cloud.dlp_v2.DlpServiceClient()

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
    bucket.delete(force=True)

def test_create_update_and_inspect_with_stored_infotype(bucket: google.cloud.storage.bucket.Bucket, capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    stored_info_type_id = ''
    try:
        create_si.create_stored_infotype(GCLOUD_PROJECT, STORED_INFO_TYPE_ID, bucket.name)
        (out, _) = capsys.readouterr()
        assert STORED_INFO_TYPE_ID in out
        stored_info_type_id = str(out).split('\n')[0].split(':')[1].strip()
        update_si.update_stored_infotype(GCLOUD_PROJECT, STORED_INFO_TYPE_ID, f'{bucket.name}/{RESOURCE_FILE_NAMES[0]}', f'{bucket.name}')
        (out, _) = capsys.readouterr()
        assert stored_info_type_id in out
        time.sleep(30)
        inspect_si.inspect_with_stored_infotype(GCLOUD_PROJECT, STORED_INFO_TYPE_ID, 'The commit was made by gary1998')
        (out, _) = capsys.readouterr()
        assert 'STORED_TYPE' in out
        assert 'Quote: gary1998' in out
    finally:
        DLP_CLIENT.delete_stored_info_type(name=stored_info_type_id)