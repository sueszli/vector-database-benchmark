import uuid
from google.api_core.retry import Retry
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import pytest
import speech_to_storage_beta
STORAGE_URI = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'
storage_client = storage.Client()
BUCKET_UUID = str(uuid.uuid4())[:8]
BUCKET_NAME = f'speech-{BUCKET_UUID}'
BUCKET_PREFIX = 'export-transcript-output-test'
DELIMETER = None
INPUT_STORAGE_URI = 'gs://cloud-samples-data/speech/commercial_mono.wav'
OUTPUT_STORAGE_URI = f'gs://{BUCKET_NAME}/{BUCKET_PREFIX}'
encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
sample_rate_hertz = 8000
language_code = 'en-US'

@Retry()
def test_export_transcript_to_storage_beta(bucket, capsys):
    if False:
        for i in range(10):
            print('nop')
    results = speech_to_storage_beta.export_transcript_to_storage_beta(INPUT_STORAGE_URI, OUTPUT_STORAGE_URI, encoding, sample_rate_hertz, language_code, BUCKET_NAME, BUCKET_PREFIX)
    assert len(results) > 0

@pytest.fixture
def bucket():
    if False:
        for i in range(10):
            print('nop')
    'Yields a bucket that is deleted after the test completes.'
    bucket = None
    while bucket is None or bucket.exists():
        bucket = storage_client.bucket(BUCKET_NAME)
    bucket.storage_class = 'COLDLINE'
    storage_client.create_bucket(bucket, location='us')
    yield bucket
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix=BUCKET_PREFIX)
    for blob in blobs:
        blob.delete()
    bucket.delete(force=True)