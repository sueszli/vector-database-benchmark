from google.api_core.retry import Retry
import pytest
import transcribe_context_classes

@Retry()
def test_transcribe_context_classes(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    result = transcribe_context_classes.transcribe_context_classes('gs://cloud-samples-data/speech/commercial_mono.wav')
    (out, _) = capsys.readouterr()
    assert 'First alternative of result ' in out
    assert result is not None