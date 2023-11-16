import uuid
from google.api_core.retry import Retry
import google.auth
from google.cloud import speech_v1p1beta1 as speech
import pytest
import speech_model_adaptation_beta
STORAGE_URI = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'
(_, PROJECT_ID) = google.auth.default()
LOCATION = 'global'
client = speech.AdaptationClient()

@Retry()
def test_model_adaptation_beta(custom_class_id: str, phrase_set_id: str, capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    class_id = custom_class_id
    phrase_id = phrase_set_id
    transcript = speech_model_adaptation_beta.transcribe_with_model_adaptation(PROJECT_ID, LOCATION, STORAGE_URI, class_id, phrase_id)
    assert 'how long is the Brooklyn Bridge' in transcript

@pytest.fixture
def custom_class_id() -> str:
    if False:
        i = 10
        return i + 15
    custom_class_id = f'customClassId{str(uuid.uuid4())[:8]}'
    yield custom_class_id
    CLASS_PARENT = f'projects/{PROJECT_ID}/locations/{LOCATION}/customClasses/{custom_class_id}'
    client.delete_custom_class(name=CLASS_PARENT)

@pytest.fixture
def phrase_set_id() -> str:
    if False:
        i = 10
        return i + 15
    phrase_set_id = f'phraseSetId{str(uuid.uuid4())[:8]}'
    yield phrase_set_id
    PHRASE_PARENT = f'projects/{PROJECT_ID}/locations/{LOCATION}/phraseSets/{phrase_set_id}'
    client.delete_phrase_set(name=PHRASE_PARENT)