from google.cloud import speech_v2

def sample_create_custom_class():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.CreateCustomClassRequest(parent='parent_value')
    operation = client.create_custom_class(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)