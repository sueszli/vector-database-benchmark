import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_word_time_offsets_v2(project_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        for i in range(10):
            print('nop')
    'Transcribe an audio file.'
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long', features=cloud_speech.RecognitionFeatures(enable_word_time_offsets=True))
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, content=content)
    response = client.recognize(request=request)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_word_time_offsets_v2(args.project_id, args.audio_file)