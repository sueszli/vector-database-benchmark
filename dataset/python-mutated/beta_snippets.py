"""Google Cloud Speech API sample that demonstrates enhanced models
and recognition metadata.

Example usage:
    python beta_snippets.py enhanced-model
    python beta_snippets.py metadata
    python beta_snippets.py punctuation
    python beta_snippets.py diarization
    python beta_snippets.py multi-channel
    python beta_snippets.py multi-language
    python beta_snippets.py word-level-conf
    python beta_snippets.py spoken-punctuation-emojis
"""
import argparse
from google.cloud import speech_v1p1beta1 as speech

def transcribe_file_with_enhanced_model() -> speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe the given audio file using an enhanced model.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', use_enhanced=True, model='phone_call')
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
    return response.results

def transcribe_file_with_metadata() -> speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Send a request that includes recognition metadata.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    metadata = speech.RecognitionMetadata()
    metadata.interaction_type = speech.RecognitionMetadata.InteractionType.DISCUSSION
    metadata.microphone_distance = speech.RecognitionMetadata.MicrophoneDistance.NEARFIELD
    metadata.recording_device_type = speech.RecognitionMetadata.RecordingDeviceType.SMARTPHONE
    metadata.recording_device_name = 'Pixel 2 XL'
    metadata.industry_naics_code_of_audio = 519190
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', metadata=metadata)
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
    return response.results

def transcribe_file_with_auto_punctuation() -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribe the given audio file with auto punctuation enabled.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', enable_automatic_punctuation=True)
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
    return response.results

def transcribe_file_with_diarization() -> speech.RecognizeResponse:
    if False:
        i = 10
        return i + 15
    'Transcribe the given audio file synchronously with diarization.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    diarization_config = speech.SpeakerDiarizationConfig(enable_speaker_diarization=True, min_speaker_count=2, max_speaker_count=10)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', diarization_config=diarization_config)
    print('Waiting for operation to complete...')
    response = client.recognize(config=config, audio=audio)
    result = response.results[-1]
    words_info = result.alternatives[0].words
    for word_info in words_info:
        print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")
    return result

def transcribe_file_with_multichannel() -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribe the given audio file synchronously with\n    multi channel.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/Google_Gnome.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US', audio_channel_count=1, enable_separate_recognition_per_channel=True)
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
        print(f'Channel Tag: {result.channel_tag}')
    return response.results

def transcribe_file_with_multilanguage() -> speech.RecognizeResponse:
    if False:
        i = 10
        return i + 15
    'Transcribe the given audio file synchronously with\n    multi language.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/multi.wav'
    first_lang = 'en-US'
    second_lang = 'es'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=44100, audio_channel_count=2, language_code=first_lang, alternative_language_codes=[second_lang])
    print('Waiting for operation to complete...')
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}: {alternative}')
        print(f'Transcript: {alternative.transcript}')
    return response.results

def transcribe_file_with_word_level_confidence() -> speech.RecognizeResponse:
    if False:
        while True:
            i = 10
    'Transcribe the given audio file synchronously with\n    word level confidence.'
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()
    speech_file = 'resources/Google_Gnome.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=16000, language_code='en-US', enable_word_confidence=True)
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
        print('First Word and Confidence: ({}, {})'.format(alternative.words[0].word, alternative.words[0].confidence))
    return response.results

def transcribe_file_with_spoken_punctuation_end_emojis() -> speech.RecognizeResponse:
    if False:
        print('Hello World!')
    'Transcribe the given audio file with spoken punctuation and emojis enabled.'
    from google.cloud import speech_v1p1beta1 as speech
    from google.protobuf import wrappers_pb2
    client = speech.SpeechClient()
    speech_file = 'resources/commercial_mono.wav'
    with open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=8000, language_code='en-US', enable_spoken_punctuation=wrappers_pb2.BoolValue(value=True), enable_spoken_emojis=wrappers_pb2.BoolValue(value=True))
    response = client.recognize(config=config, audio=audio)
    for (i, result) in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print(f'First alternative of result {i}')
        print(f'Transcript: {alternative.transcript}')
    return response.results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command')
    args = parser.parse_args()
    if args.command == 'enhanced-model':
        transcribe_file_with_enhanced_model()
    elif args.command == 'metadata':
        transcribe_file_with_metadata()
    elif args.command == 'punctuation':
        transcribe_file_with_auto_punctuation()
    elif args.command == 'diarization':
        transcribe_file_with_diarization()
    elif args.command == 'multi-channel':
        transcribe_file_with_multichannel()
    elif args.command == 'multi-language':
        transcribe_file_with_multilanguage()
    elif args.command == 'word-level-conf':
        transcribe_file_with_word_level_confidence()
    elif args.command == 'spoken-punctuation-emojis':
        transcribe_file_with_spoken_punctuation_end_emojis()