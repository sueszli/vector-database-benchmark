import os
from uuid import uuid4
from google.api_core.retry import Retry
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import pytest
import create_recognizer

def delete_recognizer(name: str) -> None:
    if False:
        while True:
            i = 10
    client = SpeechClient()
    request = cloud_speech.DeleteRecognizerRequest(name=name)
    client.delete_recognizer(request=request)

@Retry()
def test_create_recognizer(capsys: pytest.CaptureFixture, request: pytest.FixtureRequest) -> None:
    if False:
        while True:
            i = 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    recognizer_id = 'recognizer-' + str(uuid4())

    def cleanup():
        if False:
            i = 10
            return i + 15
        delete_recognizer(f'projects/{project_id}/locations/global/recognizers/{recognizer_id}')
    request.addfinalizer(cleanup)
    recognizer = create_recognizer.create_recognizer(project_id, recognizer_id)
    assert recognizer_id in recognizer.name