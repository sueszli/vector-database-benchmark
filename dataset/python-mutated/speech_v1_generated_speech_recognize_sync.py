from google.cloud import speech_v1

def sample_recognize():
    if False:
        while True:
            i = 10
    client = speech_v1.SpeechClient()
    config = speech_v1.RecognitionConfig()
    config.language_code = 'language_code_value'
    audio = speech_v1.RecognitionAudio()
    audio.content = b'content_blob'
    request = speech_v1.RecognizeRequest(config=config, audio=audio)
    response = client.recognize(request=request)
    print(response)