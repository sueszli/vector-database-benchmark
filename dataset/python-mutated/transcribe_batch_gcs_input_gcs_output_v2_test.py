import os
import re
from uuid import uuid4
from flaky import flaky
from google.cloud import storage
import pytest
import transcribe_batch_gcs_input_gcs_output_v2
_TEST_AUDIO_FILE_PATH = 'gs://cloud-samples-data/speech/audio.flac'

@pytest.fixture
def gcs_bucket() -> str:
    if False:
        return 10
    client = storage.Client()
    bucket = client.bucket('speech-samples-' + str(uuid4()))
    new_bucket = client.create_bucket(bucket, location='us')
    yield new_bucket.name
    bucket.delete(force=True)

@flaky(max_runs=10, min_passes=1)
def test_transcribe_batch_gcs_input_gcs_output_v2(gcs_bucket: pytest.CaptureFixture, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_batch_gcs_input_gcs_output_v2.transcribe_batch_gcs_input_gcs_output_v2(project_id, _TEST_AUDIO_FILE_PATH, f'gs://{gcs_bucket}')
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)