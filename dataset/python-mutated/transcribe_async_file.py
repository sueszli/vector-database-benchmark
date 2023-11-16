"""Google Cloud Speech-to-Text sample application using gRPC for async
batch processing.
"""
from google.cloud import speech

def transcribe_file(speech_file: str) -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribe the given audio file asynchronously.'
    client = speech.SpeechClient()
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    '\n     Note that transcription is limited to a 60 seconds audio file.\n     Use a GCS file for audio longer than 1 minute.\n    '
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US')
    operation = client.long_running_recognize(config=config, audio=audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
        print(f'Confidence: {result.alternatives[0].confidence}')
    return response