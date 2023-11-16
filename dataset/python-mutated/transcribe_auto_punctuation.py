"""Google Cloud Speech API sample that demonstrates auto punctuation
and recognition metadata.

Example usage:
    python transcribe_auto_punctuation.py resources/commercial_mono.wav
"""
import argparse
from google.cloud import speech

def transcribe_file_with_auto_punctuation(path: str) -> speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe the given audio file with auto punctuation enabled.'
    client = speech.SpeechClient()
    with open(path, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', enable_automatic_punctuation=True)
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
    transcribe_file_with_auto_punctuation(args.path)