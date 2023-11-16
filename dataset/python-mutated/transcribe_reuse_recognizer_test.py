import os
import re
from uuid import uuid4
from google.api_core.retry import Retry
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import pytest
import transcribe_reuse_recognizer
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

def create_recognizer(project_id: str, recognizer_id: str) -> None:
    if False:
        while True:
            i = 10
    client = SpeechClient()
    request = cloud_speech.CreateRecognizerRequest(parent=f'projects/{project_id}/locations/global', recognizer_id=recognizer_id, recognizer=cloud_speech.Recognizer(default_recognition_config=cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')))
    client.create_recognizer(request=request)

def delete_recognizer(project_id: str, recognizer_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    client = SpeechClient()
    request = cloud_speech.DeleteRecognizerRequest(name=f'projects/{project_id}/locations/global/recognizers/{recognizer_id}')
    client.delete_recognizer(request=request)

@Retry()
def test_transcribe_reuse_recognizer(capsys: pytest.CaptureFixture, request: pytest.FixtureRequest) -> None:
    if False:
        print('Hello World!')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    recognizer_id = 'recognizer-' + str(uuid4())

    def cleanup():
        if False:
            while True:
                i = 10
        delete_recognizer(project_id, recognizer_id)
    request.addfinalizer(cleanup)
    create_recognizer(project_id, recognizer_id)
    response = transcribe_reuse_recognizer.transcribe_reuse_recognizer(project_id, recognizer_id, os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)