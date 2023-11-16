import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_gcs_v2
_TEST_AUDIO_FILE_PATH = 'gs://cloud-samples-data/speech/audio.flac'

@Retry()
def test_transcribe_gcs_v2(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_gcs_v2.transcribe_gcs_v2(project_id, _TEST_AUDIO_FILE_PATH)
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)