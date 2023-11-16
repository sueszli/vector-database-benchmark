import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_reuse_recognizer(project_id: str, recognizer_id: str, audio_file: str) -> cloud_speech.RecognizeResponse:
    if False:
        return 10
    'Transcribe an audio file using an existing recognizer.'
    client = SpeechClient()
    with open(audio_file, 'rb') as f:
        content = f.read()
    request = cloud_speech.RecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/{recognizer_id}', content=content)
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
    transcribe_reuse_recognizer(args.project_id, args.audio_file)