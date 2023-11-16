"""Google Cloud Speech API sample application using the REST API for batch
processing.

Example usage:
    python transcribe.py resources/audio.raw
    python transcribe.py gs://cloud-samples-tests/speech/brooklyn.flac
"""
import argparse
from google.cloud import speech

def transcribe_file(speech_file: str) -> speech.RecognizeResponse:
    if False:
        i = 10
        return i + 15
    'Transcribe the given audio file.'
    client = speech.SpeechClient()
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US')
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response

def transcribe_gcs(gcs_uri: str) -> speech.RecognizeResponse:
    if False:
        i = 10
        return i + 15
    'Transcribes the audio file specified by the gcs_uri.'
    from google.cloud import speech
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.FLAC, sample_rate_hertz=16000, language_code='en-US')
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='File or GCS path for audio file to be recognized')
    args = parser.parse_args()
    if args.path.startswith('gs://'):
        transcribe_gcs(args.path)
    else:
        transcribe_file(args.path)