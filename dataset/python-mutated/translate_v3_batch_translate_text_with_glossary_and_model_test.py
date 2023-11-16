import os
import uuid
from google.cloud import storage
import pytest
import translate_v3_batch_translate_text_with_glossary_and_model
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
GLOSSARY_ID = 'DO_NOT_DELETE_TEST_GLOSSARY'
MODEL_ID = 'TRL251293382528204800'

@pytest.fixture(scope='function')
def bucket() -> storage.Bucket:
    if False:
        return 10
    'Create a temporary bucket to store annotation output.'
    bucket_name = 'test-bucket-for-glossary-' + str(uuid.uuid1())
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    yield bucket
    bucket.delete(force=True)

def test_batch_translate_text_with_glossary_and_model(capsys: pytest.LogCaptureFixture, bucket: storage.Bucket) -> None:
    if False:
        for i in range(10):
            print('nop')
    response = translate_v3_batch_translate_text_with_glossary_and_model.batch_translate_text_with_glossary_and_model('gs://cloud-samples-data/translation/text_with_custom_model_and_glossary.txt', f'gs://{bucket.name}/translation/BATCH_TRANSLATION_GLOS_MODEL_OUTPUT/', PROJECT_ID, MODEL_ID, GLOSSARY_ID)
    (out, _) = capsys.readouterr()
    assert 'Total Characters: 25' in out
    assert response is not None