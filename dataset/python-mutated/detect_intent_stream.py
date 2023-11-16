"""DialogFlow API Detect Intent Python sample with audio files processed as an audio stream.

Examples:
  python detect_intent_stream.py -h
  python detect_intent_stream.py --agent AGENT   --session-id SESSION_ID --audio-file-path resources/hello.wav
"""
import argparse
import uuid
from google.cloud.dialogflowcx_v3beta1.services.agents import AgentsClient
from google.cloud.dialogflowcx_v3beta1.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3beta1.types import audio_config
from google.cloud.dialogflowcx_v3beta1.types import session

def run_sample():
    if False:
        return 10
    project_id = 'YOUR-PROJECT-ID'
    location_id = 'YOUR-LOCATION-ID'
    agent_id = 'YOUR-AGENT-ID'
    agent = f'projects/{project_id}/locations/{location_id}/agents/{agent_id}'
    session_id = uuid.uuid4()
    audio_file_path = 'YOUR-AUDIO-FILE-PATH'
    language_code = 'en-us'
    detect_intent_stream(agent, session_id, audio_file_path, language_code)

def detect_intent_stream(agent, session_id, audio_file_path, language_code):
    if False:
        while True:
            i = 10
    'Returns the result of detect intent with streaming audio as input.\n\n    Using the same `session_id` between requests allows continuation\n    of the conversation.'
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
    input_audio_config = audio_config.InputAudioConfig(audio_encoding=audio_config.AudioEncoding.AUDIO_ENCODING_LINEAR_16, sample_rate_hertz=24000)

    def request_generator():
        if False:
            while True:
                i = 10
        audio_input = session.AudioInput(config=input_audio_config)
        query_input = session.QueryInput(audio=audio_input, language_code=language_code)
        voice_selection = audio_config.VoiceSelectionParams()
        synthesize_speech_config = audio_config.SynthesizeSpeechConfig()
        output_audio_config = audio_config.OutputAudioConfig()
        voice_selection.name = 'en-GB-Standard-A'
        voice_selection.ssml_gender = audio_config.SsmlVoiceGender.SSML_VOICE_GENDER_FEMALE
        synthesize_speech_config.voice = voice_selection
        output_audio_config.audio_encoding = audio_config.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_UNSPECIFIED
        output_audio_config.synthesize_speech_config = synthesize_speech_config
        yield session.StreamingDetectIntentRequest(session=session_path, query_input=query_input, output_audio_config=output_audio_config)
        with open(audio_file_path, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                audio_input = session.AudioInput(audio=chunk)
                query_input = session.QueryInput(audio=audio_input)
                yield session.StreamingDetectIntentRequest(query_input=query_input)
    responses = session_client.streaming_detect_intent(requests=request_generator())
    print('=' * 20)
    for response in responses:
        print(f'Intermediate transcript: "{response.recognition_result.transcript}".')
    response = response.detect_intent_response
    print(f'Query text: {response.query_result.transcript}')
    response_messages = [' '.join(msg.text.text) for msg in response.query_result.response_messages]
    print(f"Response text: {' '.join(response_messages)}\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', help='Agent resource name.  Required.', required=True)
    parser.add_argument('--session-id', help='Identifier of the DetectIntent session. Defaults to a random UUID.', default=str(uuid.uuid4()))
    parser.add_argument('--language-code', help='Language code of the query. Defaults to "en-US".', default='en-US')
    parser.add_argument('--audio-file-path', help='Path to the audio file.', required=True)
    args = parser.parse_args()
    detect_intent_stream(args.agent, args.session_id, args.audio_file_path, args.language_code)