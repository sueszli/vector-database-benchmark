"""Cloud Media Translation sample application.

Example usage:
    python translate_from_file.py resources/audio.raw
"""
from google.cloud import mediatranslation

def translate_from_file(file_path='path/to/your/file'):
    if False:
        while True:
            i = 10
    client = mediatranslation.SpeechTranslationServiceClient()
    audio_config = mediatranslation.TranslateSpeechConfig(audio_encoding='linear16', source_language_code='en-US', target_language_code='fr-FR')
    streaming_config = mediatranslation.StreamingTranslateSpeechConfig(audio_config=audio_config, single_utterance=True)

    def request_generator(config, audio_file_path):
        if False:
            print('Hello World!')
        yield mediatranslation.StreamingTranslateSpeechRequest(streaming_config=config)
        with open(audio_file_path, 'rb') as audio:
            while True:
                chunk = audio.read(4096)
                if not chunk:
                    break
                yield mediatranslation.StreamingTranslateSpeechRequest(audio_content=chunk)
    requests = request_generator(streaming_config, file_path)
    responses = client.streaming_translate_speech(requests)
    for response in responses:
        print(f'Response: {response}')
        result = response.result
        translation = result.text_translation_result.translation
        if result.text_translation_result.is_final:
            print(f'\nFinal translation: {translation}')
            break
        print(f'\nPartial translation: {translation}')