import os
import re
from google.api_core.retry import Retry
import change_speech_v2_location
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_change_speech_v2_location() -> None:
    if False:
        print('Hello World!')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = change_speech_v2_location.change_speech_v2_location(project_id, 'us-central1', os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)