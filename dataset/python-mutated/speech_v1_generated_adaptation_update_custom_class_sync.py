from google.cloud import speech_v1

def sample_update_custom_class():
    if False:
        while True:
            i = 10
    client = speech_v1.AdaptationClient()
    request = speech_v1.UpdateCustomClassRequest()
    response = client.update_custom_class(request=request)
    print(response)