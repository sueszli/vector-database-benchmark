"""Google Cloud Speech-to-Text sample application using the gRPC for async
batch processing.
"""

def transcribe_gcs(gcs_uri: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Asynchronously transcribes the audio file specified by the gcs_uri.\n\n    Args:\n        gcs_uri: The Google Cloud Storage path to an audio file.\n\n    Returns:\n        The generated transcript from the audio file provided.\n    '
    from google.cloud import speech
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.FLAC, sample_rate_hertz=44100, language_code='en-US')
    operation = client.long_running_recognize(config=config, audio=audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)
    transcript_builder = []
    for result in response.results:
        transcript_builder.append(f'\nTranscript: {result.alternatives[0].transcript}')
        transcript_builder.append(f'\nConfidence: {result.alternatives[0].confidence}')
    transcript = ''.join(transcript_builder)
    print(transcript)
    return transcript