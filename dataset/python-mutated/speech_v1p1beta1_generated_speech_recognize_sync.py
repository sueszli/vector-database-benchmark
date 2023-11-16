from google.cloud import speech_v1p1beta1

def sample_recognize():
    if False:
        i = 10
        return i + 15
    client = speech_v1p1beta1.SpeechClient()
    config = speech_v1p1beta1.RecognitionConfig()
    config.language_code = 'language_code_value'
    audio = speech_v1p1beta1.RecognitionAudio()
    audio.content = b'content_blob'
    request = speech_v1p1beta1.RecognizeRequest(config=config, audio=audio)
    response = client.recognize(request=request)
    print(response)