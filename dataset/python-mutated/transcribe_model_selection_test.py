import os
import re
from google.api_core.retry import Retry
import pytest
import transcribe_model_selection
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_model_selection_file(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    response = transcribe_model_selection.transcribe_model_selection(os.path.join(RESOURCES, 'Google_Gnome.wav'), 'video')
    (out, err) = capsys.readouterr()
    assert re.search('the weather outside is sunny', out, re.DOTALL | re.I)
    assert response is not None

@Retry()
def test_transcribe_model_selection_gcs(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    response = transcribe_model_selection.transcribe_model_selection_gcs('gs://cloud-samples-tests/speech/Google_Gnome.wav', 'video')
    (out, err) = capsys.readouterr()
    assert re.search('the weather outside is sunny', out, re.DOTALL | re.I)
    assert response is not None