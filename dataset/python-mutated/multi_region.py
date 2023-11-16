from google.api_core import client_options
from google.cloud import speech

def sync_recognize_with_multi_region_gcs() -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Recognizes speech synchronously in the GCS bucket.'
    _client_options = client_options.ClientOptions(api_endpoint='eu-speech.googleapis.com')
    client = speech.SpeechClient(client_options=_client_options)
    gcs_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.raw'
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US')
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response.results
sync_recognize_with_multi_region_gcs()