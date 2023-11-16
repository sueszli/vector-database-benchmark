from google.cloud import speech_v1p1beta1

def sample_long_running_recognize():
    if False:
        print('Hello World!')
    client = speech_v1p1beta1.SpeechClient()
    config = speech_v1p1beta1.RecognitionConfig()
    config.language_code = 'language_code_value'
    audio = speech_v1p1beta1.RecognitionAudio()
    audio.content = b'content_blob'
    request = speech_v1p1beta1.LongRunningRecognizeRequest(config=config, audio=audio)
    operation = client.long_running_recognize(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)