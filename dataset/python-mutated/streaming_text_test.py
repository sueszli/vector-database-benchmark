import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import streaming_text
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_streaming_prediction() -> None:
    if False:
        for i in range(10):
            print('nop')
    responses = streaming_text.streaming_prediction(project_id=_PROJECT_ID, location=_LOCATION)
    print(responses)
    assert '1.' in responses
    assert '?' in responses
    assert 'you' in responses
    assert 'do' in responses