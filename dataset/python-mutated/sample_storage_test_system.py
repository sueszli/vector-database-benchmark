from datetime import datetime
from os import getenv, path
import subprocess
import time
import uuid
from google.cloud import storage
import pytest
PROJECT = getenv('GCP_PROJECT')
BUCKET = getenv('BUCKET')
assert PROJECT is not None
assert BUCKET is not None

@pytest.fixture(scope='module')
def storage_client():
    if False:
        print('Hello World!')
    yield storage.Client()

@pytest.fixture(scope='module')
def bucket_object(storage_client):
    if False:
        return 10
    bucket_object = storage_client.get_bucket(BUCKET)
    yield bucket_object

@pytest.fixture(scope='module')
def uploaded_file(bucket_object):
    if False:
        while True:
            i = 10
    name = f'test-{str(uuid.uuid4())}.txt'
    blob = bucket_object.blob(name)
    test_dir = path.dirname(path.abspath(__file__))
    blob.upload_from_filename(path.join(test_dir, 'test.txt'))
    yield name
    blob.delete()

def test_hello_gcs(uploaded_file):
    if False:
        while True:
            i = 10
    start_time = datetime.utcnow().isoformat()
    time.sleep(10)
    log_process = subprocess.Popen(['gcloud', 'alpha', 'functions', 'logs', 'read', 'hello_gcs_generic', '--start-time', start_time], stdout=subprocess.PIPE)
    logs = str(log_process.communicate()[0])
    assert uploaded_file in logs