import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def adaptation_v2_custom_class_reference(project_id: str, phrase_set_id: str, custom_class_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        for i in range(10):
            print('nop')
    'Transcribe audio file using a custom class.\n\n    Args:\n        project_id: The GCP project ID.\n        phrase_set_id: The ID of the phrase set to use.\n        custom_class_id: The ID of the custom class to use.\n        audio_file: The audio file to transcribe.\n\n    Returns:\n        The transcript of the audio file.\n    '
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    request = cloud_speech.CreateCustomClassRequest(parent=f'projects/{project_id}/locations/global', custom_class_id=custom_class_id, custom_class=cloud_speech.CustomClass(items=[{'value': 'fare'}]))
    operation = client.create_custom_class(request=request)
    custom_class = operation.result()
    request = cloud_speech.CreatePhraseSetRequest(parent=f'projects/{project_id}/locations/global', phrase_set_id=phrase_set_id, phrase_set=cloud_speech.PhraseSet(phrases=[{'value': f'${{{custom_class.name}}}', 'boost': 20}]))
    operation = client.create_phrase_set(request=request)
    phrase_set = operation.result()
    adaptation = cloud_speech.SpeechAdaptation(phrase_sets=[cloud_speech.SpeechAdaptation.AdaptationPhraseSet(phrase_set=phrase_set.name)])
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), adaptation=adaptation, language_codes=['en-US'], model='short')
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, content=content)
    response = client.recognize(request=request)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('phrase_set_id', help='ID for the phrase set to create')
    parser.add_argument('custom_class_id', help='ID for the custom class to create')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    adaptation_v2_custom_class_reference(args.project_id, args.phrase_set_id, args.custom_class_id, args.audio_file)