"""Google Cloud Text-To-Speech API sample application.

Example usage:
    python list_voices.py
"""

def list_voices():
    if False:
        for i in range(10):
            print('nop')
    'Lists the available voices.'
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices()
    for voice in voices.voices:
        print(f'Name: {voice.name}')
        for language_code in voice.language_codes:
            print(f'Supported language: {language_code}')
        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender)
        print(f'SSML Voice Gender: {ssml_gender.name}')
        print(f'Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}\n')
if __name__ == '__main__':
    list_voices()