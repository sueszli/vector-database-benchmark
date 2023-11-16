import os
import re
from google.api_core.retry import Retry
import transcribe_multiple_languages_v2
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_multiple_languages_v2() -> None:
    if False:
        print('Hello World!')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_multiple_languages_v2.transcribe_multiple_languages_v2(project_id, ['en-US', 'fr-FR'], os.path.join(RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)