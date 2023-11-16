import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def adaptation_v2_inline_phrase_set(project_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        print('Hello World!')
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    phrase_set = cloud_speech.PhraseSet(phrases=[{'value': 'fare', 'boost': 10}])
    adaptation = cloud_speech.SpeechAdaptation(phrase_sets=[cloud_speech.SpeechAdaptation.AdaptationPhraseSet(inline_phrase_set=phrase_set)])
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), adaptation=adaptation, language_codes=['en-US'], model='short')
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
    adaptation_v2_inline_phrase_set(args.project_id, args.audio_file)