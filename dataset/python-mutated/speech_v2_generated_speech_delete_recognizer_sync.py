from google.cloud import speech_v2

def sample_delete_recognizer():
    if False:
        while True:
            i = 10
    client = speech_v2.SpeechClient()
    request = speech_v2.DeleteRecognizerRequest(name='name_value')
    operation = client.delete_recognizer(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)