from google.cloud import speech

def transcribe_diarization_gcs_beta(gcs_uri: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Transcribe a remote audio file (stored in Google Cloud Storage) using speaker diarization.\n\n    Args:\n        gcs_uri: The Google Cloud Storage path to an audio file.\n\n    Returns:\n        True if the operation successfully completed, False otherwise.\n    '
    client = speech.SpeechClient()
    speaker_diarization_config = speech.SpeakerDiarizationConfig(enable_speaker_diarization=True, min_speaker_count=2, max_speaker_count=2)
    recognition_config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, language_code='en-US', sample_rate_hertz=8000, diarization_config=speaker_diarization_config)
    audio = speech.RecognitionAudio(uri=gcs_uri)
    response = client.long_running_recognize(config=recognition_config, audio=audio).result(timeout=300)
    result = response.results[-1]
    words_info = result.alternatives[0].words
    for word_info in words_info:
        print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")
    return True