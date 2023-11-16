from google.cloud import speech_v2

def sample_get_config():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.GetConfigRequest(name='name_value')
    response = client.get_config(request=request)
    print(response)