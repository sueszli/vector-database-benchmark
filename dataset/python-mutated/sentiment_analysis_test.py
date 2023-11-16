import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import sentiment_analysis
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_sentiment_analysis() -> None:
    if False:
        return 10
    content = sentiment_analysis.sentiment_analysis(temperature=0, project_id=_PROJECT_ID, location=_LOCATION)
    assert content is not None