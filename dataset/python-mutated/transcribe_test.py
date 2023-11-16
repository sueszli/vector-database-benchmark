import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_file(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    transcribe.transcribe_file(os.path.join(RESOURCES, 'audio.raw'))
    (out, err) = capsys.readouterr()
    assert re.search('how old is the Brooklyn Bridge', out, re.DOTALL | re.I)

@Retry()
def test_transcribe_gcs(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    transcribe.transcribe_gcs('gs://python-docs-samples-tests/speech/audio.flac')
    (out, err) = capsys.readouterr()
    assert re.search('how old is the Brooklyn Bridge', out, re.DOTALL | re.I)