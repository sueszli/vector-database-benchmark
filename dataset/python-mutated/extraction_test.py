import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import extraction
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_extractive_question_answering() -> None:
    if False:
        return 10
    content = extraction.extractive_question_answering(temperature=0, project_id=_PROJECT_ID, location=_LOCATION)
    assert content == 'Reduced moist tropical vegetation cover in the basin.'