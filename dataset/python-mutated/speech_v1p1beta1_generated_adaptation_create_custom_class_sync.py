from google.cloud import speech_v1p1beta1

def sample_create_custom_class():
    if False:
        while True:
            i = 10
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.CreateCustomClassRequest(parent='parent_value', custom_class_id='custom_class_id_value')
    response = client.create_custom_class(request=request)
    print(response)