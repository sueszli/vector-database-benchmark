from google.cloud import texttospeech_v1beta1

def sample_synthesize_long_audio():
    if False:
        return 10
    client = texttospeech_v1beta1.TextToSpeechLongAudioSynthesizeClient()
    input = texttospeech_v1beta1.SynthesisInput()
    input.text = 'text_value'
    audio_config = texttospeech_v1beta1.AudioConfig()
    audio_config.audio_encoding = 'ALAW'
    voice = texttospeech_v1beta1.VoiceSelectionParams()
    voice.language_code = 'language_code_value'
    request = texttospeech_v1beta1.SynthesizeLongAudioRequest(input=input, audio_config=audio_config, output_gcs_uri='output_gcs_uri_value', voice=voice)
    operation = client.synthesize_long_audio(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)