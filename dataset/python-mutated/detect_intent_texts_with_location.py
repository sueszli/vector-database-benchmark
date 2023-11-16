"""DialogFlow API Detect Intent Python sample to use regional endpoint.

Examples:
  python detect_intent_texts_with_location.py -h
  python detect_intent_texts_with_location.py --project-id PROJECT_ID   --location-id LOCATION_ID --session-id SESSION_ID   "hello" "book a meeting room" "Mountain View"
"""
import argparse
import uuid

def detect_intent_texts_with_location(project_id, location_id, session_id, texts, language_code):
    if False:
        i = 10
        return i + 15
    'Returns the result of detect intent with texts as inputs.\n\n    Using the same `session_id` between requests allows continuation\n    of the conversation.'
    from google.cloud import dialogflow
    session_client = dialogflow.SessionsClient(client_options={'api_endpoint': f'{location_id}-dialogflow.googleapis.com'})
    session = f'projects/{project_id}/locations/{location_id}/agent/sessions/{session_id}'
    print(f'Session path: {session}\n')
    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        response = session_client.detect_intent(request={'session': session, 'query_input': query_input})
        print('=' * 20)
        print(f'Query text: {response.query_result.query_text}')
        print(f'Detected intent: {response.query_result.intent.display_name} (confidence: {(response.query_result.intent_detection_confidence,)})\n')
        print(f'Fulfillment text: {response.query_result.fulfillment_text}\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', help='Project/agent id.  Required.', required=True)
    parser.add_argument('--location-id', help='Location id.  Required.', required=True)
    parser.add_argument('--session-id', help='Identifier of the DetectIntent session. Defaults to a random UUID.', default=str(uuid.uuid4()))
    parser.add_argument('--language-code', help='Language code of the query. Defaults to "en-US".', default='en-US')
    parser.add_argument('texts', nargs='+', type=str, help='Text inputs.')
    args = parser.parse_args()
    detect_intent_texts_with_location(args.project_id, args.location_id, args.session_id, args.texts, args.language_code)