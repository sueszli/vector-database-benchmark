import os
import re
from google.api_core.retry import Retry
import transcribe_multichannel_v2
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_multichannel_v2() -> None:
    if False:
        for i in range(10):
            print('nop')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    response = transcribe_multichannel_v2.transcribe_multichannel_v2(project_id, os.path.join(_RESOURCES, 'two_channel_16k.wav'))
    assert re.search('saving account', response.results[0].alternatives[0].transcript, re.DOTALL | re.I)
    assert response.results[0].channel_tag == 1
    assert re.search('debit card number', response.results[1].alternatives[0].transcript, re.DOTALL | re.I)
    assert response.results[1].channel_tag == 2