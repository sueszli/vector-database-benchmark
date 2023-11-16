from google.cloud import speech_v2

def sample_undelete_custom_class():
    if False:
        return 10
    client = speech_v2.SpeechClient()
    request = speech_v2.UndeleteCustomClassRequest(name='name_value')
    operation = client.undelete_custom_class(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)