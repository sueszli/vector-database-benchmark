import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def create_recognizer(project_id: str, recognizer_id: str) -> cloud_speech.Recognizer:
    if False:
        i = 10
        return i + 15
    client = SpeechClient()
    request = cloud_speech.CreateRecognizerRequest(parent=f'projects/{project_id}/locations/global', recognizer_id=recognizer_id, recognizer=cloud_speech.Recognizer(default_recognition_config=cloud_speech.RecognitionConfig(language_codes=['en-US'], model='long')))
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()
    print('Created Recognizer:', recognizer.name)
    return recognizer
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('recognizer_id', help='ID for the recognizer to create')
    args = parser.parse_args()
    create_recognizer()