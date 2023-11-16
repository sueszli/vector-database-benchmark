import os
import uuid
from google.cloud import storage
import pytest
import translate_v3_batch_translate_text_with_model
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
MODEL_ID = 'TRL251293382528204800'

@pytest.fixture(scope='function')
def bucket() -> storage.Bucket:
    if False:
        i = 10
        return i + 15
    'Create a temporary bucket to store annotation output.'
    bucket_name = f'tmp-{uuid.uuid4().hex}'
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    yield bucket
    bucket.delete(force=True)

def test_batch_translate_text_with_model(capsys: pytest.LogCaptureFixture, bucket: storage.Bucket) -> None:
    if False:
        return 10
    response = translate_v3_batch_translate_text_with_model.batch_translate_text_with_model('gs://cloud-samples-data/translation/custom_model_text.txt', f'gs://{bucket.name}/translation/BATCH_TRANSLATION_MODEL_OUTPUT/', PROJECT_ID, MODEL_ID)
    (out, _) = capsys.readouterr()
    assert 'Total Characters: 15' in out
    assert 'Translated Characters: 15' in out
    assert response is not None