import os
import re
from google.api_core.retry import Retry
import adaptation_v2_inline_custom_class
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_adaptation_v2_inline_custom_class() -> None:
    if False:
        print('Hello World!')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = adaptation_v2_inline_custom_class.adaptation_v2_inline_custom_class(project_id, os.path.join(_RESOURCES, 'fair.wav'))
    assert re.search('the word', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)