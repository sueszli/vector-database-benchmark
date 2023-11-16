import os
import uuid
from google.cloud import storage
import pytest
import translate_v3_batch_translate_text
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.fixture(scope='function')
def bucket() -> storage.Bucket:
    if False:
        for i in range(10):
            print('nop')
    'Create a temporary bucket to store annotation output.'
    bucket_name = f'test-{uuid.uuid4()}'
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    yield bucket
    bucket.delete(force=True)

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_batch_translate_text(capsys: pytest.LogCaptureFixture, bucket: storage.Bucket) -> None:
    if False:
        print('Hello World!')
    response = translate_v3_batch_translate_text.batch_translate_text('gs://cloud-samples-data/translation/text.txt', f'gs://{bucket.name}/translation/BATCH_TRANSLATION_OUTPUT/', PROJECT_ID, timeout=320)
    (out, _) = capsys.readouterr()
    assert response.translated_characters is not None