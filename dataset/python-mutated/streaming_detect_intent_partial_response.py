import uuid
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.types import audio_config
from google.cloud.dialogflowcx_v3.types import InputAudioConfig
from google.cloud.dialogflowcx_v3.types import session

def run_sample():
    if False:
        return 10
    '\n    TODO(developer): Modify these variables before running the sample.\n    '
    project_id = 'YOUR-PROJECT-ID'
    location = 'YOUR-LOCATION-ID'
    agent_id = 'YOUR-AGENT-ID'
    audio_file_name = 'YOUR-AUDIO-FILE-PATH'
    encoding = 'AUDIO_ENCODING_LINEAR_16'
    sample_rate_hertz = 16000
    language_code = 'en'
    streaming_detect_intent_partial_response(project_id, location, agent_id, audio_file_name, encoding, sample_rate_hertz, language_code)

def streaming_detect_intent_partial_response(project_id, location, agent_id, audio_file_name, encoding, sample_rate_hertz, language_code):
    if False:
        while True:
            i = 10
    client_options = None
    if location != 'global':
        api_endpoint = f'{location}-dialogflow.googleapis.com:443'
        print(f'API Endpoint: {api_endpoint}\n')
        client_options = {'api_endpoint': api_endpoint}
    session_client = SessionsClient(client_options=client_options)
    session_id = str(uuid.uuid4())
    session_path = session_client.session_path(project=project_id, location=location, agent=agent_id, session=session_id)

    def request_generator():
        if False:
            return 10
        audio_encoding = audio_config.AudioEncoding[encoding]
        config = InputAudioConfig(audio_encoding=audio_encoding, sample_rate_hertz=sample_rate_hertz, single_utterance=True)
        audio_input = session.AudioInput(config=config)
        query_input = session.QueryInput(audio=audio_input, language_code=language_code)
        yield session.StreamingDetectIntentRequest(session=session_path, query_input=query_input, enable_partial_response=True)
        with open(audio_file_name, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                audio_input = session.AudioInput(audio=chunk, config=config)
                query_input = session.QueryInput(audio=audio_input, language_code=language_code)
                yield session.StreamingDetectIntentRequest(session=session_path, query_input=query_input, enable_partial_response=True)
    responses = session_client.streaming_detect_intent(requests=request_generator())
    print('=' * 20)
    for response in responses:
        print(f'Intermediate transcript: "{response.recognition_result.transcript}".')
    response = response.detect_intent_response
    print(f'Query text: {response.query_result.transcript}')
    response_messages = [' '.join(msg.text.text) for msg in response.query_result.response_messages]
    print(f"Response text: {' '.join(response_messages)}\n")
if __name__ == '__main__':
    run_sample()