from google.cloud import speech_v1p1beta1 as speech

def transcribe_file_with_multilanguage_gcs(gcs_uri: str) -> str:
    if False:
        while True:
            i = 10
    'Transcribe a remote audio file with multi-language recognition\n\n    Args:\n        gcs_uri: The Google Cloud Storage path to an audio file.\n\n    Returns:\n        The generated transcript from the audio file provided.\n    '
    client = speech.SpeechClient()
    first_language = 'ja-JP'
    alternate_languages = ['es-ES', 'en-US']
    recognition_config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code=first_language, alternative_language_codes=alternate_languages)
    audio = speech.RecognitionAudio(uri=gcs_uri)
    response = client.long_running_recognize(config=recognition_config, audio=audio).result(timeout=300)
    transcript_builder = []
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        transcript_builder.append('-' * 20)
        transcript_builder.append(f'First alternative of result {i}: {alternative}')
        transcript_builder.append(f'Transcript: {alternative.transcript}')
    transcript = ''.join(transcript_builder)
    print(transcript)
    return transcript