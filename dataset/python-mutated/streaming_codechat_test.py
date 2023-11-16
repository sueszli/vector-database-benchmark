import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import streaming_codechat
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_streaming_prediction() -> None:
    if False:
        i = 10
        return i + 15
    responses = streaming_codechat.streaming_prediction(project_id=_PROJECT_ID, location=_LOCATION)
    assert 'def' in responses