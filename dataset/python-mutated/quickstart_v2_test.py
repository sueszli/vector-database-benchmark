import os
import re
from google.api_core.retry import Retry
import quickstart_v2
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_quickstart_v2() -> None:
    if False:
        while True:
            i = 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = quickstart_v2.quickstart_v2(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    assert re.search('how old is the Brooklyn Bridge', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)