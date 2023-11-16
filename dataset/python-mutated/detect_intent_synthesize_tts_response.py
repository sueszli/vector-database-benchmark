import uuid
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.types import audio_config
from google.cloud.dialogflowcx_v3.types import session

def run_sample():
    if False:
        return 10
    project_id = 'YOUR-PROJECT-ID'
    location = 'YOUR-LOCATION-ID'
    agent_id = 'YOUR-AGENT-ID'
    text = 'YOUR-TEXT'
    audio_encoding = 'YOUR-AUDIO-ENCODING'
    language_code = 'YOUR-LANGUAGE-CODE'
    output_file = 'YOUR-OUTPUT-FILE'
    detect_intent_synthesize_tts_response(project_id, location, agent_id, text, audio_encoding, language_code, output_file)

def detect_intent_synthesize_tts_response(project_id, location, agent_id, text, audio_encoding, language_code, output_file):
    if False:
        return 10
    'Returns the result of detect intent with synthesized response.'
    client_options = None
    if location != 'global':
        api_endpoint = f'{location}-dialogflow.googleapis.com:443'
        print(f'API Endpoint: {api_endpoint}\n')
        client_options = {'api_endpoint': api_endpoint}
    session_client = SessionsClient(client_options=client_options)
    session_id = str(uuid.uuid4())
    session_path = session_client.session_path(project=project_id, location=location, agent=agent_id, session=session_id)
    text_input = session.TextInput(text=text)
    query_input = session.QueryInput(text=text_input, language_code=language_code)
    synthesize_speech_config = audio_config.SynthesizeSpeechConfig(speaking_rate=1.25, pitch=10.0)
    output_audio_config = audio_config.OutputAudioConfig(synthesize_speech_config=synthesize_speech_config, audio_encoding=audio_config.OutputAudioEncoding[audio_encoding])
    request = session.DetectIntentRequest(session=session_path, query_input=query_input, output_audio_config=output_audio_config)
    response = session_client.detect_intent(request=request)
    print(f'Speaking Rate: {response.output_audio_config.synthesize_speech_config.speaking_rate}')
    print(f'Pitch: {response.output_audio_config.synthesize_speech_config.pitch}')
    with open(output_file, 'wb') as fout:
        fout.write(response.output_audio)
    print(f'Audio content written to file: {output_file}')
if __name__ == '__main__':
    run_sample()