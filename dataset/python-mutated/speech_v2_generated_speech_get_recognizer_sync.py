from google.cloud import speech_v2

def sample_get_recognizer():
    if False:
        print('Hello World!')
    client = speech_v2.SpeechClient()
    request = speech_v2.GetRecognizerRequest(name='name_value')
    response = client.get_recognizer(request=request)
    print(response)