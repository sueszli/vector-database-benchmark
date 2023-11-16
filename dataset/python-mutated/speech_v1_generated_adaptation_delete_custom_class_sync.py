from google.cloud import speech_v1

def sample_delete_custom_class():
    if False:
        while True:
            i = 10
    client = speech_v1.AdaptationClient()
    request = speech_v1.DeleteCustomClassRequest(name='name_value')
    client.delete_custom_class(request=request)