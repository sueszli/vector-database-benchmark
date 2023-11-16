import os
from google.api_core.retry import Retry
import pytest
import transcribe_auto_punctuation
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_file_with_auto_punctuation(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    result = transcribe_auto_punctuation.transcribe_file_with_auto_punctuation('resources/commercial_mono.wav')
    (out, _) = capsys.readouterr()
    assert 'First alternative of result ' in out
    assert result is not None