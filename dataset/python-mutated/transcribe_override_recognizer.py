import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.field_mask_pb2 import FieldMask

def transcribe_override_recognizer(project_id: str, recognizer_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribe an audio file using an existing recognizer.'
    client = SpeechClient()
    request = cloud_speech.CreateRecognizerRequest(parent=f'projects/{project_id}/locations/global', recognizer_id=recognizer_id, recognizer=cloud_speech.Recognizer(default_recognition_config=cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='latest_long', features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True, enable_word_time_offsets=True))))
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()
    print('Created Recognizer:', recognizer.name)
    with open(audio_file, 'rb') as f:
        content = f.read()
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/{recognizer_id}', config=cloud_speech.RecognitionConfig(features=cloud_speech.RecognitionFeatures(enable_word_time_offsets=False)), config_mask=FieldMask(paths=['features.enable_word_time_offsets']), content=content)
    response = client.recognize(request=request)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('recognizer_id', help='Recognizer ID to use for recogniition')
    parser.add_argument('audio_file', help='Audio file to stream')
    args = parser.parse_args()
    transcribe_override_recognizer(args.project_id, args.audio_file)