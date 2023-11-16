from google.cloud import speech_v2

def sample_recognize():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v2.SpeechClient()
    request = speech_v2.RecognizeRequest(content=b'content_blob', recognizer='recognizer_value')
    response = client.recognize(request=request)
    print(response)