import argparse
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def change_speech_v2_location(project_id: str, location: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe an audio file in a specific region.'
    client = SpeechClient(client_options=ClientOptions(api_endpoint=f'{location}-speech.googleapis.com'))
    with open(audio_file, 'rb') as f:
        content = f.read()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/{location}/recognizers/_', config=config, content=content)
    response = client.recognize(request=request)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('location', help='The GCP region to which to connect')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    change_speech_v2_location(args.project_id, args.location, args.audio_file)