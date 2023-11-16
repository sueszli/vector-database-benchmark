import os
import re
from flaky import flaky
import pytest
import transcribe_batch_dynamic_batching_v2
_TEST_AUDIO_FILE_PATH = 'gs://cloud-samples-data/speech/audio.flac'

@flaky(max_runs=10, min_passes=1)
def test_transcribe_batch_dynamic_batching_v2(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_batch_dynamic_batching_v2.transcribe_batch_dynamic_batching_v2(project_id, _TEST_AUDIO_FILE_PATH)
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)