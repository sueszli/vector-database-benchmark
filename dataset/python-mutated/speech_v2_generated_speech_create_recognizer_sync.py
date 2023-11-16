from google.cloud import speech_v2

def sample_create_recognizer():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.CreateRecognizerRequest(parent='parent_value')
    operation = client.create_recognizer(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)