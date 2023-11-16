from google.ai import generativelanguage_v1beta2

def sample_get_model():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta2.ModelServiceClient()
    request = generativelanguage_v1beta2.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)