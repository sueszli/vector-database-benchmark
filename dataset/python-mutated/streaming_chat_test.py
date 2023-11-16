import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import streaming_chat
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_streaming_prediction() -> None:
    if False:
        print('Hello World!')
    responses = streaming_chat.streaming_prediction(project_id=_PROJECT_ID, location=_LOCATION)
    assert 'Earth' in responses