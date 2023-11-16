import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_async_file
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    result = transcribe_async_file.transcribe_file(os.path.join(RESOURCES, 'audio.raw'))
    (out, err) = capsys.readouterr()
    assert re.search('how old is the Brooklyn Bridge', out, re.DOTALL | re.I)
    assert result is not None