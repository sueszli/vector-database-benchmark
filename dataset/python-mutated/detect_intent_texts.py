"""DialogFlow API Detect Intent Python sample with text inputs.

Examples:
  python detect_intent_texts.py -h
  python detect_intent_texts.py --agent AGENT   --session-id SESSION_ID   "hello" "book a meeting room" "Mountain View"
  python detect_intent_texts.py --agent AGENT   --session-id SESSION_ID   "tomorrow" "10 AM" "2 hours" "10 people" "A" "yes"
"""
import argparse
import uuid
from google.cloud.dialogflowcx_v3beta1.services.agents import AgentsClient
from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3beta1.types import session

def run_sample():
    if False:
        for i in range(10):
            print('nop')
    project_id = 'YOUR-PROJECT-ID'
    location_id = 'YOUR-LOCATION-ID'
    agent_id = 'YOUR-AGENT-ID'
    agent = f'projects/{project_id}/locations/{location_id}/agents/{agent_id}'
    session_id = uuid.uuid4()
    texts = ['Hello']
    language_code = 'en-us'
    detect_intent_texts(agent, session_id, texts, language_code)

def detect_intent_texts(agent, session_id, texts, language_code):
    if False:
        return 10
    'Returns the result of detect intent with texts as inputs.\n\n    Using the same `session_id` between requests allows continuation\n    of the conversation.'
    session_path = f'{agent}/sessions/{session_id}'
    print(f'Session path: {session_path}\n')
    client_options = None
    agent_components = AgentsClient.parse_agent_path(agent)
    location_id = agent_components['location']
    if location_id != 'global':
        api_endpoint = f'{location_id}-dialogflow.googleapis.com:443'
        print(f'API Endpoint: {api_endpoint}\n')
        client_options = {'api_endpoint': api_endpoint}
    session_client = SessionsClient(client_options=client_options)
    for text in texts:
        text_input = session.TextInput(text=text)
        query_input = session.QueryInput(text=text_input, language_code=language_code)
        request = session.DetectIntentRequest(session=session_path, query_input=query_input)
        response = session_client.detect_intent(request=request)
        print('=' * 20)
        print(f'Query text: {response.query_result.text}')
        response_messages = [' '.join(msg.text.text) for msg in response.query_result.response_messages]
        print(f"Response text: {' '.join(response_messages)}\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', help='Agent resource name.  Required.', required=True)
    parser.add_argument('--session-id', help='Identifier of the DetectIntent session. Defaults to a random UUID.', default=str(uuid.uuid4()))
    parser.add_argument('--language-code', help='Language code of the query. Defaults to "en-US".', default='en-US')
    parser.add_argument('texts', nargs='+', type=str, help='Text inputs.')
    args = parser.parse_args()
    detect_intent_texts(args.agent, args.session_id, args.texts, args.language_code)