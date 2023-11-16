"""Google Cloud Speech API sample that demonstrates enhanced models
and recognition metadata.

Example usage:
    python transcribe_enhanced_model.py resources/commercial_mono.wav
"""
import argparse
from google.cloud import speech

def transcribe_file_with_enhanced_model(path: str) -> speech.RecognizeResponse:
    if False:
        return 10
    'Transcribe the given audio file using an enhanced model.'
    client = speech.SpeechClient()
    with open(path, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', use_enhanced=True, model='phone_call')
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='File to stream to the API')
    args = parser.parse_args()
    transcribe_file_with_enhanced_model(args.path)