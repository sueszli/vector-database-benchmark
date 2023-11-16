import argparse
from typing import List
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_multiple_languages_v2(project_id: str, language_codes: List[str], audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe an audio file.'
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=language_codes, model='latest_long')
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, content=content)
    response = client.recognize(request=request)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('language_codes', nargs='+', help='Language codes to transcribe')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_multiple_languages_v2(args.project_id, args.language_codes, args.audio_file)