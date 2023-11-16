import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import summarization
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'
expected_response = 'The efficient-market hypothesis'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_text_summarization() -> None:
    if False:
        for i in range(10):
            print('nop')
    content = summarization.text_summarization(temperature=0, project_id=_PROJECT_ID, location=_LOCATION)
    assert expected_response in content