import os
from google.api_core.retry import Retry
import pytest
from transcribe_multichannel import transcribe_file_with_multichannel, transcribe_gcs_with_multichannel
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_multichannel_file(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    result = transcribe_file_with_multichannel(os.path.join(RESOURCES, 'multi.wav'))
    (out, err) = capsys.readouterr()
    assert 'how are you doing' in out
    assert result is not None

@Retry()
def test_transcribe_multichannel_gcs(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_gcs_with_multichannel('gs://cloud-samples-data/speech/multi.wav')
    (out, err) = capsys.readouterr()
    assert 'how are you doing' in out
    assert result is not None