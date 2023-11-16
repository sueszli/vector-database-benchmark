from google.cloud import speech_v2

def sample_get_custom_class():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.GetCustomClassRequest(name='name_value')
    response = client.get_custom_class(request=request)
    print(response)