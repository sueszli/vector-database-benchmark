import os
import re
from google.api_core.retry import Retry
import transcribe_word_level_confidence_v2
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_word_level_confidence_v2() -> None:
    if False:
        i = 10
        return i + 15
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_word_level_confidence_v2.transcribe_word_level_confidence_v2(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)
    assert response.results[0].alternatives[0].words[0].confidence > 0