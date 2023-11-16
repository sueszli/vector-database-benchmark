from google.cloud import speech_v2

def sample_update_recognizer():
    if False:
        while True:
            i = 10
    client = speech_v2.SpeechClient()
    request = speech_v2.UpdateRecognizerRequest()
    operation = client.update_recognizer(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)