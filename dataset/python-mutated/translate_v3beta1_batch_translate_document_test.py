import os
import uuid
from google.cloud import storage
import pytest
import translate_v3beta1_batch_translate_document
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture(scope='function')
def bucket() -> storage.Bucket:
    if False:
        for i in range(10):
            print('nop')
    bucket_name = f'test-{uuid.uuid4()}'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket = storage_client.create_bucket(bucket, location='us-central1')
    yield bucket
    bucket.delete(force=True)

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_batch_translate_document(capsys: pytest.LogCaptureFixture, bucket: storage.Bucket) -> None:
    if False:
        while True:
            i = 10
    response = translate_v3beta1_batch_translate_document.batch_translate_document(input_uri='gs://cloud-samples-data/translation/async_invoices/*', output_uri=f'gs://{bucket.name}/translation/BATCH_TRANSLATE_DOCUMENT_OUTPUT/', project_id=PROJECT_ID, timeout=1000)
    (out, _) = capsys.readouterr()
    assert 'Total Pages' in out
    assert response.total_pages is not None