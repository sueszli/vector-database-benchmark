"""Dialogflow API Beta Detect Intent Python sample with an audio response.

Examples:
  python detect_intent_with_texttospeech_response_test.py -h
  python detect_intent_with_texttospeech_response_test.py   --project-id PROJECT_ID --session-id SESSION_ID "hello"
"""
import argparse
import uuid

def detect_intent_with_texttospeech_response(project_id, session_id, texts, language_code):
    if False:
        for i in range(10):
            print('nop')
    'Returns the result of detect intent with texts as inputs and includes\n    the response in an audio format.\n\n    Using the same `session_id` between requests allows continuation\n    of the conversation.'
    from google.cloud import dialogflow
    session_client = dialogflow.SessionsClient()
    session_path = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session_path))
    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        output_audio_config = dialogflow.OutputAudioConfig(audio_encoding=dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16)
        request = dialogflow.DetectIntentRequest(session=session_path, query_input=query_input, output_audio_config=output_audio_config)
        response = session_client.detect_intent(request=request)
        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(response.query_result.intent.display_name, response.query_result.intent_detection_confidence))
        print('Fulfillment text: {}\n'.format(response.query_result.fulfillment_text))
        with open('output.wav', 'wb') as out:
            out.write(response.output_audio)
            print('Audio content written to file "output.wav"')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', help='Project/agent id.  Required.', required=True)
    parser.add_argument('--session-id', help='Identifier of the DetectIntent session. Defaults to a random UUID.', default=str(uuid.uuid4()))
    parser.add_argument('--language-code', help='Language code of the query. Defaults to "en-US".', default='en-US')
    parser.add_argument('texts', nargs='+', type=str, help='Text inputs.')
    args = parser.parse_args()
    detect_intent_with_texttospeech_response(args.project_id, args.session_id, args.texts, args.language_code)