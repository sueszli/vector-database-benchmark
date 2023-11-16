from google.cloud import speech_v1p1beta1 as speech

def transcribe_file_with_word_level_confidence(gcs_uri: str) -> str:
    if False:
        print('Hello World!')
    'Transcribe a remote audio file with word level confidence.\n\n    Args:\n        gcs_uri: The Google Cloud Storage path to an audio file.\n\n    Returns:\n        The generated transcript from the audio file provided.\n    '
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.FLAC, sample_rate_hertz=44100, language_code='en-US', enable_word_confidence=True)
    audio = speech.RecognitionAudio(uri=gcs_uri)
    response = client.long_running_recognize(config=config, audio=audio).result(timeout=300)
    transcript_builder = []
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        transcript_builder.append('-' * 20)
        transcript_builder.append(f'\nFirst alternative of result {i}')
        transcript_builder.append(f'\nTranscript: {alternative.transcript}')
        transcript_builder.append('\nFirst Word and Confidence: ({}, {})'.format(alternative.words[0].word, alternative.words[0].confidence))
    transcript = ''.join(transcript_builder)
    print(transcript)
    return transcript