from google.cloud import speech_v2

def sample_create_phrase_set():
    if False:
        print('Hello World!')
    client = speech_v2.SpeechClient()
    request = speech_v2.CreatePhraseSetRequest(parent='parent_value')
    operation = client.create_phrase_set(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)