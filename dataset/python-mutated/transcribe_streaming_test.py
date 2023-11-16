import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_streaming
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_streaming(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    transcribe_streaming.transcribe_streaming(os.path.join(RESOURCES, 'audio.raw'))
    (out, _) = capsys.readouterr()
    assert re.search('how old is the Brooklyn Bridge', out, re.DOTALL | re.I)