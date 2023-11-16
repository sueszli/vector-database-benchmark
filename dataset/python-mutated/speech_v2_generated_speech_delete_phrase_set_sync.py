from google.cloud import speech_v2

def sample_delete_phrase_set():
    if False:
        while True:
            i = 10
    client = speech_v2.SpeechClient()
    request = speech_v2.DeletePhraseSetRequest(name='name_value')
    operation = client.delete_phrase_set(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)