from google.cloud import speech_v2

def sample_undelete_phrase_set():
    if False:
        return 10
    client = speech_v2.SpeechClient()
    request = speech_v2.UndeletePhraseSetRequest(name='name_value')
    operation = client.undelete_phrase_set(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)