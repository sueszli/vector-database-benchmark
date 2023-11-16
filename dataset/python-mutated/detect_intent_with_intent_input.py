"""DialogFlow Detect Intent Python sample with specified intent."""
import uuid
from google.cloud.dialogflowcx_v3.services.intents import IntentsClient
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.types import session

def run_sample():
    if False:
        while True:
            i = 10
    project_id = 'YOUR-PROJECT-ID'
    location = 'YOUR-LOCATION-ID'
    agent_id = 'YOUR-AGENT-ID'
    intent_id = 'YOUR-INTENT-ID'
    language_code = 'en-us'
    detect_intent_with_intent_input(project_id, location, agent_id, intent_id, language_code)

def detect_intent_with_intent_input(project_id, location, agent_id, intent_id, language_code):
    if False:
        for i in range(10):
            print('nop')
    'Returns the result of detect intent with sentiment analysis'
    client_options = None
    if location != 'global':
        api_endpoint = f'{location}-dialogflow.googleapis.com:443'
        print(f'API Endpoint: {api_endpoint}\n')
        client_options = {'api_endpoint': api_endpoint}
    session_client = SessionsClient(client_options=client_options)
    session_id = str(uuid.uuid4())
    intents_client = IntentsClient()
    session_path = session_client.session_path(project=project_id, location=location, agent=agent_id, session=session_id)
    intent_path = intents_client.intent_path(project=project_id, location=location, agent=agent_id, intent=intent_id)
    intent = session.IntentInput(intent=intent_path)
    query_input = session.QueryInput(intent=intent, language_code=language_code)
    request = session.DetectIntentRequest(session=session_path, query_input=query_input)
    response = session_client.detect_intent(request=request)
    response_text = []
    for response_message in response.query_result.response_messages:
        response_text.append(response_message.text.text)
        print(response_message.text.text)
    return response_text
if __name__ == '__main__':
    run_sample()