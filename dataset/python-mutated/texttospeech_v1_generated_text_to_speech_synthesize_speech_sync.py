from google.cloud import texttospeech_v1

def sample_synthesize_speech():
    if False:
        for i in range(10):
            print('nop')
    client = texttospeech_v1.TextToSpeechClient()
    input = texttospeech_v1.SynthesisInput()
    input.text = 'text_value'
    voice = texttospeech_v1.VoiceSelectionParams()
    voice.language_code = 'language_code_value'
    audio_config = texttospeech_v1.AudioConfig()
    audio_config.audio_encoding = 'ALAW'
    request = texttospeech_v1.SynthesizeSpeechRequest(input=input, voice=voice, audio_config=audio_config)
    response = client.synthesize_speech(request=request)
    print(response)