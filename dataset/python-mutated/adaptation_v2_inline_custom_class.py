import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def adaptation_v2_inline_custom_class(project_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe audio file using inline custom class\n\n    Args:\n        project_id: The GCP project ID.\n        audio_file: The audio file to transcribe.\n\n    Returns:\n        The response from the recognizer.\n    '
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    phrase_set = cloud_speech.PhraseSet(phrases=[{'value': '${fare}', 'boost': 20}])
    custom_class = cloud_speech.CustomClass(name='fare', items=[{'value': 'fare'}])
    adaptation = cloud_speech.SpeechAdaptation(phrase_sets=[cloud_speech.SpeechAdaptation.AdaptationPhraseSet(inline_phrase_set=phrase_set)], custom_classes=[custom_class])
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
    adaptation_v2_inline_custom_class(args.project_id, args.audio_file)