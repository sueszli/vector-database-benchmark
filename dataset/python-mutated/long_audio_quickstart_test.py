import os
import uuid
from google.cloud import storage
import pytest
from long_audio_quickstart import synthesize_long_audio
PROJECT_NUMBER = os.environ['GOOGLE_CLOUD_PROJECT_NUMBER']

@pytest.fixture(scope='module')
def test_bucket():
    if False:
        print('Hello World!')
    'Yields a bucket that is deleted after the test completes.'
    bucket = None
    while bucket is None or bucket.exists():
        bucket_name = f'tts-long-audio-test-{uuid.uuid4()}'
        bucket = storage.Client().bucket(bucket_name)
    bucket.create()
    yield bucket
    bucket.delete(force=True)

def test_synthesize_long_audio(capsys, test_bucket):
    if False:
        for i in range(10):
            print('nop')
    file_name = 'fake_file.wav'
    output_gcs_uri = f'gs://{test_bucket.name}/{file_name}'
    synthesize_long_audio(str(PROJECT_NUMBER), 'us-central1', output_gcs_uri)
    (out, _) = capsys.readouterr()
    assert 'Finished processing, check your GCS bucket to find your audio file!' in out