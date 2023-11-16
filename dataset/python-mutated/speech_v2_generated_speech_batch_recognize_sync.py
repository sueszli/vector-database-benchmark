from google.cloud import speech_v2

def sample_batch_recognize():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.BatchRecognizeRequest(recognizer='recognizer_value')
    operation = client.batch_recognize(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)