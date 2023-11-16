from google.cloud import speech_v2

def sample_delete_custom_class():
    if False:
        print('Hello World!')
    client = speech_v2.SpeechClient()
    request = speech_v2.DeleteCustomClassRequest(name='name_value')
    operation = client.delete_custom_class(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)