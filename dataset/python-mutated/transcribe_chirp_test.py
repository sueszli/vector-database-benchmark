import os
import re
from google.api_core.retry import Retry
import transcribe_chirp
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_chirp() -> None:
    if False:
        for i in range(10):
            print('nop')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_chirp.transcribe_chirp(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)