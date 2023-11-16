from google.cloud import speech_v1p1beta1

def sample_update_custom_class():
    if False:
        return 10
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.UpdateCustomClassRequest()
    response = client.update_custom_class(request=request)
    print(response)