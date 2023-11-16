from google.cloud import speech_v2

def sample_update_config():
    if False:
        i = 10
        return i + 15
    client = speech_v2.SpeechClient()
    request = speech_v2.UpdateConfigRequest()
    response = client.update_config(request=request)
    print(response)