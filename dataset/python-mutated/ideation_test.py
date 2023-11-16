import os
import backoff
from google.api_core.exceptions import ResourceExhausted
import ideation
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'
interview_expected_response = '1. What is your experience with project management?\n2. What is your process for managing a project?\n3. How do you handle unexpected challenges or roadblocks?\n4. How do you communicate with stakeholders?\n5. How do you measure the success of a project?\n6. What are your strengths and weaknesses as a project manager?\n7. What are your salary expectations?\n8. What are your career goals?\n9. Why are you interested in this position?\n10. What questions do you have for me?'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_interview() -> None:
    if False:
        print('Hello World!')
    content = ideation.interview(temperature=0, project_id=_PROJECT_ID, location=_LOCATION)
    assert content == interview_expected_response