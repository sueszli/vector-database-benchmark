import os
import re
import pytest
import transcribe_file_v2
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

def test_transcribe_file_v2(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_file_v2.transcribe_file_v2(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)