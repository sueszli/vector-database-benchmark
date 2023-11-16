import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_streaming_v2
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_streaming_v2(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    responses = transcribe_streaming_v2.transcribe_streaming_v2(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    transcript = ''
    for response in responses:
        for result in response.results:
            transcript += result.alternatives[0].transcript
    assert re.search('how old is the Brooklyn Bridge', transcript, re.DOTALL | re.I)