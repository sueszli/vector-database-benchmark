"""DialogFlow Detects intent using EventInput."""
import uuid
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.types import session

def run_sample():
    if False:
        print('Hello World!')
    project_id = 'dialogflow-cx-demo-1-348717'
    location = 'global'
    agent_id = '8caa6b47-5dd7-4380-b86e-ea4301d565b0'
    event = 'sys.no-match-default'
    language_code = 'en-us'
    detect_intent_with_event_input(project_id, location, agent_id, event, language_code)

def detect_intent_with_event_input(project_id, location, agent_id, event, language_code):
    if False:
        print('Hello World!')
    'Detects intent using EventInput'
    client_options = None
    if location != 'global':
        api_endpoint = f'{location}-dialogflow.googleapis.com:443'
        print(f'API Endpoint: {api_endpoint}\n')
        client_options = {'api_endpoint': api_endpoint}
    session_client = SessionsClient(client_options=client_options)
    session_id = str(uuid.uuid4())
    session_path = session_client.session_path(project=project_id, location=location, agent=agent_id, session=session_id)
    event = session.EventInput(event=event)
    query_input = session.QueryInput(event=event, language_code=language_code)
    request = session.DetectIntentRequest(session=session_path, query_input=query_input)
    response = session_client.detect_intent(request=request)
    response_text = response.query_result.response_messages[0].text.text[0]
    print(f'Response: {response_text}')
    return response_text
if __name__ == '__main__':
    run_sample()