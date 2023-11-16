import os
import re
from uuid import uuid4
from flaky import flaky
from google.cloud import storage
from google.cloud.speech_v2.types import cloud_speech
import pytest
import transcribe_batch_multiple_files_v2
_TEST_AUDIO_FILE_PATH = 'gs://cloud-samples-data/speech/audio.flac'
_GCS_BUCKET_OBJECT_RE = 'gs://([^/]+)/(.*)'

@pytest.fixture
def gcs_bucket() -> str:
    if False:
        while True:
            i = 10
    client = storage.Client()
    bucket = client.bucket('speech-samples-' + str(uuid4()))
    new_bucket = client.create_bucket(bucket, location='us')
    yield new_bucket.name
    bucket.delete(force=True)

def get_gcs_object(gcs_path: str) -> cloud_speech.BatchRecognizeResults:
    if False:
        for i in range(10):
            print('nop')
    client = storage.Client()
    (bucket_name, object_name) = re.match(_GCS_BUCKET_OBJECT_RE, gcs_path).group(1, 2)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    results_bytes = blob.download_as_bytes()
    return cloud_speech.BatchRecognizeResults.from_json(results_bytes, ignore_unknown_fields=True)

@flaky(max_runs=10, min_passes=1)
def test_transcribe_batch_multiple_files_v2(gcs_bucket: pytest.FixtureRequest, capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_batch_multiple_files_v2.transcribe_batch_multiple_files_v2(project_id, [_TEST_AUDIO_FILE_PATH], f'gs://{gcs_bucket}')
    results = get_gcs_object(response.results[_TEST_AUDIO_FILE_PATH].uri)
    assert re.search('how old is the Brooklyn Bridge', results.results[0].alternatives[0].transcript, re.DOTALL | re.I)