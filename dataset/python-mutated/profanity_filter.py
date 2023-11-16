""" Google Cloud Speech API sample application using the REST API for batch
processing.

Example usage:
    python transcribe.py gs://cloud-samples-tests/speech/brooklyn.flac
"""
from google.cloud import speech

def sync_recognize_with_profanity_filter_gcs(gcs_uri: str) -> speech.RecognizeResponse:
    if False:
        return 10
    client = speech.SpeechClient()
    audio = {'uri': gcs_uri}
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.FLAC, sample_rate_hertz=16000, language_code='en-US', profanity_filter=True)
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        alternative = result.alternatives[0]
        print(f'Transcript: {alternative.transcript}')
    return response.results
sync_recognize_with_profanity_filter_gcs('gs://cloud-samples-tests/speech/brooklyn.flac')