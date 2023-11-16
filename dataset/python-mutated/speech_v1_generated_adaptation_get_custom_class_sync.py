from google.cloud import speech_v1

def sample_get_custom_class():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v1.AdaptationClient()
    request = speech_v1.GetCustomClassRequest(name='name_value')
    response = client.get_custom_class(request=request)
    print(response)