import os
import re
from uuid import uuid4
import backoff
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import adaptation_v2_phrase_set_reference
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

def delete_phrase_set(name: str) -> None:
    if False:
        return 10
    client = SpeechClient()
    request = cloud_speech.DeletePhraseSetRequest(name=name)
    client.delete_phrase_set(request=request)

@backoff.on_exception(backoff.expo, Exception, max_time=120)
def test_adaptation_v2_phrase_set_reference() -> None:
    if False:
        return 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    phrase_set_id = 'phrase-set-' + str(uuid4())
    response = adaptation_v2_phrase_set_reference.adaptation_v2_phrase_set_reference(project_id, phrase_set_id, os.path.join(_RESOURCES, 'fair.wav'))
    assert re.search('the word is fare', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)
    delete_phrase_set(f'projects/{project_id}/locations/global/phraseSets/{phrase_set_id}')