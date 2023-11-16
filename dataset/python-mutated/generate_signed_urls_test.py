import os
from google.cloud import storage
import pytest
import requests
import generate_signed_urls
BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
GOOGLE_APPLICATION_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

@pytest.fixture
def test_blob():
    if False:
        for i in range(10):
            print('nop')
    'Provides a pre-existing blob in the test bucket.'
    bucket = storage.Client().bucket(BUCKET)
    blob = bucket.blob('storage_snippets_test_sigil')
    blob.upload_from_string("Hello, is it me you're looking for?")
    return blob

def test_generate_get_signed_url(test_blob, capsys):
    if False:
        return 10
    get_signed_url = generate_signed_urls.generate_signed_url(service_account_file=GOOGLE_APPLICATION_CREDENTIALS, bucket_name=BUCKET, object_name=test_blob.name, expiration=60)
    response = requests.get(get_signed_url)
    assert response.ok