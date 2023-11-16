from google.cloud import speech_v2

def sample_undelete_recognizer():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v2.SpeechClient()
    request = speech_v2.UndeleteRecognizerRequest(name='name_value')
    operation = client.undelete_recognizer(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)